from __future__ import annotations

import asyncio
import hashlib
import json
import time
import httpx

from llm.session_store import SessionStore
from logger_config import logger
from modules.critic.agent import CriticAgent
from modules.js_checker.module import JSCheckerModule
from modules.judge.agent import JudgeAgent
from modules.renderer.module import RendererModule
from modules.scene_coder.agent import SceneCoderAgent
from modules.scene_planner.agent import ScenePlannerAgent
from modules.scene_planner.schema import OSD
from pipeline.bus import EventBus
from pipeline.events import (
    CheckerFailed,
    CheckerOk,
    CoderDone,
    CriticDone,
    Event,
    PatcherDone,
    PlannerDone,
    RenderDone,
    TaskCreated,
    TaskDone,
    TaskFailed,
)
from pipeline.task import Candidate, PipelineTask
from utils.http import download_image


class Orchestrator:
    """Wires actors into the bus, enforces deadlines + iter caps, threads
    one PipelineTask through every stage."""

    def __init__(
        self,
        *,
        bus: EventBus,
        session_store: SessionStore,
        planner: ScenePlannerAgent | None,
        coder: SceneCoderAgent,
        critic: CriticAgent,
        judge: JudgeAgent | None,
        js_checker: JSCheckerModule,
        renderer: RendererModule,
        http_client: httpx.AsyncClient,
        coder_multimodal: bool = False,
        task_deadline_s: float = 60.0,
        max_iter: int = 2,
        score_threshold: float = 0.80,
        planner_workers: int = 2,
        coder_workers: int = 2,
        checker_workers: int = 2,
        renderer_workers: int = 1,
        critic_workers: int = 3,
        patcher_workers: int = 2,
        queue_size: int = 8,
        coder_ensemble_size: int = 1,
        coder_ensemble_temperature: float = 0.3,
    ) -> None:
        self.bus = bus
        self.session_store = session_store
        self.planner = planner
        self.coder = coder
        self.critic = critic
        self.judge = judge
        self.js_checker = js_checker
        self.renderer = renderer
        self.http_client = http_client
        self.coder_multimodal = coder_multimodal
        self.task_deadline_s = task_deadline_s
        self.max_iter = max_iter
        self.score_threshold = score_threshold
        self._coder_ensemble_size = max(1, coder_ensemble_size)
        self._coder_ensemble_temperature = coder_ensemble_temperature

        self._mg_render_sem = asyncio.Semaphore(max(1, renderer_workers))
        self._mg_check_sem = asyncio.Semaphore(max(1, checker_workers))
        self._total_stages = 7 if planner is not None else 6

        self._coder_stage = 2 if planner is not None else 1
        self._checker_stage = 3 if planner is not None else 2
        self._renderer_stage = 4 if planner is not None else 3
        self._critic_stage = 5 if planner is not None else 4
        self._patcher_stage = 6 if planner is not None else 5
        self._terminal_stage = 7 if planner is not None else 6
        self._tasks: dict[str, PipelineTask] = {}
        self._futures: dict[str, asyncio.Future[PipelineTask]] = {}

        self._wire_actors(
            planner_workers=planner_workers,
            coder_workers=coder_workers,
            checker_workers=checker_workers,
            renderer_workers=renderer_workers,
            critic_workers=critic_workers,
            patcher_workers=patcher_workers,
            queue_size=queue_size,
        )

    def _stg(self, n: int, name: str) -> str:
        return f"[{n}/{self._total_stages} {name}]"

    # Wiring

    def _wire_actors(
        self, *, planner_workers, coder_workers, checker_workers,
        renderer_workers, critic_workers, patcher_workers, queue_size,
    ) -> None:

        # Register actors
        if self.planner is not None:
            self.bus.register_actor("planner", self._on_task_created, workers=planner_workers, queue_size=queue_size)
        self.bus.register_actor("coder",   self._on_coder_input,  workers=coder_workers,   queue_size=queue_size)
        self.bus.register_actor("checker", self._on_coder_done,   workers=checker_workers, queue_size=queue_size)
        self.bus.register_actor("renderer",self._on_checker_ok,   workers=renderer_workers,queue_size=queue_size)
        self.bus.register_actor("critic",  self._on_render_done,  workers=critic_workers,  queue_size=queue_size)
        self.bus.register_actor("patcher", self._on_critic_done,  workers=patcher_workers, queue_size=queue_size)
        self.bus.register_actor("terminal",self._on_terminal,     workers=1,               queue_size=queue_size)

        # Subscribe to events
        if self.planner is not None:
            self.bus.subscribe("task.created", "planner")
            self.bus.subscribe("planner.done", "coder")
        else:
            # No planner — Coder is the entry point and must consume the image directly.
            self.bus.subscribe("task.created", "coder")
        self.bus.subscribe("coder.done",     "checker")
        self.bus.subscribe("checker.ok",     "renderer")
        self.bus.subscribe("checker.failed", "coder")
        self.bus.subscribe("render.done",    "critic")
        self.bus.subscribe("critic.done",    "patcher")
        self.bus.subscribe("patcher.done",   "checker")
        self.bus.subscribe("task.done",      "terminal")
        self.bus.subscribe("task.failed",    "terminal")

    # Lifecycle

    async def start(self) -> None:
        await self.bus.start()

    async def stop(self) -> None:
        await self.bus.stop()
        for task in list(self._tasks.values()):
            if task.deadline_task and not task.deadline_task.done():
                task.deadline_task.cancel()
        for stem, future in list(self._futures.items()):
            if not future.done():
                future.cancel()
            self._futures.pop(stem, None)

    # API

    async def submit(self, task: PipelineTask) -> asyncio.Future[PipelineTask]:
        """Register a PipelineTask and kick off the pipeline.

        Returns a `Future` that resolves to the same task envelope once
        it reaches a terminal state (`task.done` or `task.failed`). The
        envelope is mutated in-place, so callers that already retain a
        reference can observe the same downstream state directly.
        """
        future: asyncio.Future[PipelineTask] = asyncio.get_running_loop().create_future()
        self._futures[task.stem] = future
        self._tasks[task.stem] = task
        task.started_at = time.time()
        task.deadline_task = asyncio.create_task(
            self._deadline_watcher(task.stem),
            name=f"deadline.{task.stem}",
        )
        entry_label = 'Planner' if self.planner is not None else 'Coder'
        logger.info(
            f"{self._stg(1, entry_label)} Submitted Task {task.stem} | Seed: {task.seed} | URL: {task.image_url[:80]}"
        )

        try:
            t_fetch = time.time()
            task.image_bytes, task.image_mime = await download_image(
                task.image_url, self.http_client,
            )
            logger.info(
                f"{self._stg(1, entry_label)} Downloaded Task {task.stem} | MIME: {task.image_mime} | Bytes: {len(task.image_bytes)} | Elapsed: {time.time() - t_fetch:.2f}s"
            )
        except Exception as exc:
            await self._fail(task.stem, f"fetch: {type(exc).__name__}: {exc}", stage="fetch")
            return future
        await self.bus.publish(TaskCreated(task_id=task.stem, image_url=task.image_url, seed=task.seed))
        return future

    # Handlers

    async def _on_task_created(self, event: Event) -> None:
        """Planner-on path. Coder-direct path skips this handler entirely."""
        assert isinstance(event, TaskCreated)
        task = self._tasks.get(event.task_id)
        if task is None or task.terminal or self.planner is None:
            return
        try:
            osd = await self.planner.plan(
                task_id=task.stem,
                image_bytes=task.image_bytes,
                image_url=task.image_url,
                mime=task.image_mime,
            )
            task.osd = osd.model_dump_json(indent=2)
            await self.bus.publish(PlannerDone(
                task_id=task.stem, osd=osd.model_dump(),
            ))
        except Exception as exc:
            await self._fail(task.stem, f"planner: {type(exc).__name__}: {exc}", stage="planner")

    async def _on_coder_input(self, event: Event) -> None:
        """Handles task.created (planner-off fresh), planner.done (planner-on
        fresh), and checker.failed (repair, both modes)."""
        task = self._tasks.get(event.task_id)
        if task is None or task.terminal:
            return

        if self.planner is not None and task.osd is None:
            return
        is_fresh = isinstance(event, (TaskCreated, PlannerDone))
        # Multi-gen path — only at fresh code, only when ensemble_size > 1.
        if is_fresh and self._coder_ensemble_size > 1:
            await self._run_multi_gen_fresh(task, event)
            return
        try:
            osd = OSD.model_validate_json(task.osd) if task.osd else None
            _mode = "repair" if isinstance(event, CheckerFailed) else "fresh code"
            logger.info(f"{self._stg(self._coder_stage, 'Coder')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration} | Mode: {_mode}")
            if isinstance(event, TaskCreated):
                # Planner-off path: coder must consume the image directly.
                js_code = await self.coder.code(
                    task_id=task.stem,
                    image_bytes=task.image_bytes,
                    image_url=task.image_url,
                    image_mime=task.image_mime,
                )
            elif isinstance(event, PlannerDone):
                js_code = await self.coder.code(
                    task_id=task.stem, osd=osd,
                    image_bytes=task.image_bytes if self.coder_multimodal else None,
                    image_mime=task.image_mime,
                )
            elif isinstance(event, CheckerFailed):
                js_code = await self.coder.code_repair(
                    task_id=task.stem, osd=osd, js_errors=list(event.js_errors),
                )
            else:
                logger.warning(f"coder handler got unexpected event type={event.type}")
                return
            task.js_code = js_code
            await self.bus.publish(CoderDone(task_id=task.stem, js_code=js_code))
        except Exception as exc:
            await self._fail(task.stem, f"coder: {type(exc).__name__}: {exc}", stage="coder")


    async def _run_multi_gen_fresh(self, task: PipelineTask, event: Event) -> None:
        """Parallel best-of-K Coder with bracket-judge selection."""
        try:
            osd = OSD.model_validate_json(task.osd) if task.osd else None
            K = self._coder_ensemble_size
            base_seed = task.seed
            temperature = self._coder_ensemble_temperature
            send_image = (osd is None) or self.coder_multimodal
            logger.info(
                f"{self._stg(self._coder_stage, 'Coder')} Event: {event.type} | "
                f"Task {task.stem} | Iter: {task.iteration} | "
                f"Mode: bracket-judge K={K} (temperature={temperature})"
            )

            # 1) Generate K candidates IN PARALLEL.
            async def _gen(k: int) -> Candidate:
                cand = Candidate(k=k, seed=base_seed + k)
                t0 = time.time()
                try:
                    cand.js_code = await self.coder.code(
                        task_id=task.stem,
                        osd=osd,
                        image_bytes=task.image_bytes if send_image else None,
                        image_url=(task.image_url
                                   if (send_image and task.image_bytes is None)
                                   else None),
                        image_mime=task.image_mime,
                        candidate_id=k,
                        seed_override=cand.seed,
                        temperature_override=temperature,
                    )
                except Exception as exc:
                    cand.drop_reason = f"coder:{type(exc).__name__}"
                cand.elapsed_s = time.time() - t0
                return cand

            candidates = await asyncio.gather(*[_gen(k) for k in range(K)])

            unique_leaders: dict[str, Candidate] = {}
            duplicates: list[Candidate] = []
            for cand in candidates:
                if cand.js_code is None:
                    continue
                digest = hashlib.sha256(cand.js_code.encode("utf-8")).hexdigest()
                if digest in unique_leaders:
                    duplicates.append(cand)
                    cand.drop_reason = f"duplicate_of_k{unique_leaders[digest].k}"
                else:
                    unique_leaders[digest] = cand
            if duplicates:
                logger.info(
                    f"{self._stg(self._coder_stage, 'Coder')} Dedupe Task {task.stem} | "
                    f"K={K} | Unique={len(unique_leaders)} | "
                    f"Duplicates={[(c.k, c.drop_reason) for c in duplicates]}"
                )

            # 2) Validate each UNIQUE candidate IN PARALLEL (Checker + Renderer).
            async def _validate(cand: Candidate) -> Candidate:
                if cand.js_code is None:
                    return cand
                shadow = PipelineTask(
                    stem=f"{task.stem}#k{cand.k}", image_url=task.image_url,
                )
                shadow.js_code = cand.js_code
                shadow.image_bytes = task.image_bytes
                shadow.image_mime = task.image_mime
                t_eval = time.time()
                async with self._mg_check_sem:
                    try:
                        await self.js_checker.process(shadow)
                    except Exception as exc:
                        cand.drop_reason = f"checker:{type(exc).__name__}"
                        cand.elapsed_s += time.time() - t_eval
                        return cand
                cand.js_valid = shadow.js_valid
                cand.js_errors = list(shadow.js_errors or [])
                if not shadow.js_valid:
                    cand.drop_reason = "checker"
                    cand.elapsed_s += time.time() - t_eval
                    return cand
                async with self._mg_render_sem:
                    try:
                        await self.renderer.process(shadow)
                    except Exception as exc:
                        cand.drop_reason = f"renderer:{type(exc).__name__}"
                        cand.elapsed_s += time.time() - t_eval
                        return cand
                cand.rendered_png = shadow.rendered_png
                cand.render_errors = list(shadow.render_errors or [])
                if not shadow.rendered_png:
                    cand.drop_reason = "renderer"
                cand.elapsed_s += time.time() - t_eval
                return cand

            # Run validation only on unique js_code; copy results to dupes.
            unique_list = list(unique_leaders.values())
            unique_list = await asyncio.gather(*[_validate(c) for c in unique_list])
            # Re-index after gather (gather preserves order, but be explicit).
            for cand in unique_list:
                digest = hashlib.sha256(cand.js_code.encode("utf-8")).hexdigest()
                unique_leaders[digest] = cand
            for dup in duplicates:
                digest = hashlib.sha256(dup.js_code.encode("utf-8")).hexdigest()
                leader = unique_leaders[digest]
                # Copy validation/render results from leader; keep drop_reason
                # so the bracket excludes duplicates from pairwise matches.
                dup.js_valid = leader.js_valid
                dup.js_errors = list(leader.js_errors)
                dup.rendered_png = leader.rendered_png
                dup.render_errors = list(leader.render_errors)
            task.candidates = list(candidates)

            survivors = [
                c for c in candidates
                if c.drop_reason is None and c.rendered_png is not None
            ]
            drops = [c.drop_reason for c in candidates]
            elapsed = [round(c.elapsed_s, 1) for c in candidates]

            if not survivors:
                logger.warning(
                    f"{self._stg(self._coder_stage, 'Coder')} Bracket Task {task.stem} | "
                    f"K={K} | Drops={drops} | Elapsed={elapsed} | ALL CANDIDATES FAILED"
                )
                await self._fail(
                    task.stem, "multi-gen: all candidates failed", stage="coder",
                )
                return

            # 3) Bracket — pairwise judge until a single winner remains.
            winner = await self._run_bracket(
                task=task, survivors=survivors,
            )

            logger.info(
                f"{self._stg(self._coder_stage, 'Coder')} Bracket Task {task.stem} | "
                f"K={K} | Survivors={[c.k for c in survivors]} | "
                f"Drops={drops} | Elapsed={elapsed} | Bracket winner=k{winner.k}"
            )

            # 4) Run Critic ONCE on the bracket winner to populate
            #    CriticReport (the repair loop reads issues / matching_aspects).
            artifact_context = {
                "kind": "coder_v1",
                "js_code": winner.js_code,
                "osd": json.loads(task.osd) if task.osd else None,
            }
            try:
                winner.critic_report = await self.critic.critique(
                    task_id=f"{task.stem}#k{winner.k}",
                    image_bytes=task.image_bytes,
                    image_mime=task.image_mime,
                    render_png=winner.rendered_png,
                    artifact_context=artifact_context,
                )
            except Exception as exc:
                await self._fail(
                    task.stem,
                    f"multi-gen: critic on winner failed: {type(exc).__name__}: {exc}",
                    stage="critic",
                )
                return

            # 5) Promote winner's coder session, evict losers, hand off to repair.
            task.winner_k = winner.k
            self.session_store.rename_actor(task.stem, f"coder#k{winner.k}", "coder")
            for c in candidates:
                if c.k != winner.k:
                    self.session_store.evict_actor(task.stem, f"coder#k{c.k}")
            task.js_code = winner.js_code
            task.js_valid = winner.js_valid
            task.js_errors = list(winner.js_errors)
            task.rendered_png = winner.rendered_png
            await self.bus.publish(CriticDone(
                task_id=task.stem, report=winner.critic_report,
            ))
        except Exception as exc:
            await self._fail(
                task.stem, f"multi-gen: {type(exc).__name__}: {exc}", stage="coder",
            )

    async def _run_bracket(
        self, *, task: PipelineTask, survivors: list[Candidate],
    ) -> Candidate:
        """Single-elimination bracket using the Judge."""
        if len(survivors) == 1:
            return survivors[0]
        if self.judge is None:
            logger.warning(
                f"[Bracket] Task {task.stem} | judge is None, falling back to k{survivors[0].k}"
            )
            return survivors[0]

        round_idx = 1
        current = list(survivors)
        while len(current) > 1:
            pairs: list[tuple[Candidate, Candidate]] = []
            byes: list[Candidate] = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    pairs.append((current[i], current[i + 1]))
                else:
                    byes.append(current[i])

            async def _match(pair_idx: int, a: Candidate, b: Candidate) -> Candidate:
                label = f"R{round_idx}M{pair_idx + 1} k{a.k}-vs-k{b.k}"
                verdict = await self.judge.compare(
                    task_id=task.stem,
                    match_label=label,
                    reference_bytes=task.image_bytes,
                    reference_mime=task.image_mime,
                    render_a=a.rendered_png,
                    render_b=b.rendered_png,
                )
                return a if verdict.winner == "A" else b

            results = await asyncio.gather(*[
                _match(idx, a, b) for idx, (a, b) in enumerate(pairs)
            ])
            current = list(results) + byes
            logger.info(
                f"[Bracket R{round_idx}] Task {task.stem} | "
                f"matches={len(pairs)} | byes={[c.k for c in byes]} | "
                f"advancing={[c.k for c in current]}"
            )
            round_idx += 1
        return current[0]

    async def _on_coder_done(self, event: Event) -> None:
        """Consumes coder.done and patcher.done. Runs JS Checker on task.js_code."""
        task = self._tasks.get(event.task_id)
        if task is None or task.terminal:
            return
        logger.info(f"{self._stg(self._checker_stage, 'Checker')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration}")
        if not isinstance(task.js_code, str) or not task.js_code:
            await self._fail(task.stem, "coder returned empty js_code", stage="coder")
            return
        # Reset checker output before re-running.
        task.js_valid = None
        task.js_errors = []
        task.failed = False
        task.failure_reason = None
        try:
            await self.js_checker.process(task)
        except Exception as exc:
            await self._fail(task.stem, f"checker: {type(exc).__name__}: {exc}", stage="checker")
            return
        if task.js_valid:
            await self.bus.publish(CheckerOk(
                task_id=task.stem, js_code=task.js_code,
                metrics=dict(task.js_metrics or {}),
            ))
        else:
            await self.bus.publish(CheckerFailed(
                task_id=task.stem, js_errors=list(task.js_errors or []),
            ))

    async def _on_checker_ok(self, event: Event) -> None:
        assert isinstance(event, CheckerOk)
        task = self._tasks.get(event.task_id)
        if task is None or task.terminal:
            return
        logger.info(f"{self._stg(self._renderer_stage, 'Renderer')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration}")

        task.render_errors = []
        task.failed = False
        task.failure_reason = None
        try:
            await self.renderer.process(task)
        except Exception as exc:
            await self._fail(task.stem, f"renderer: {type(exc).__name__}: {exc}", stage="renderer")
            return
        if task.failed or task.rendered_png is None:
            reason = task.failure_reason or (task.render_errors[0] if task.render_errors else "no png")
            await self._fail(task.stem, f"renderer: {reason}", stage="renderer")
            return
        await self.bus.publish(RenderDone(
            task_id=task.stem, rendered_png=task.rendered_png,
        ))

    async def _on_render_done(self, event: Event) -> None:
        assert isinstance(event, RenderDone)
        task = self._tasks.get(event.task_id)
        if task is None or task.terminal:
            return
        logger.info(f"{self._stg(self._critic_stage, 'Critic')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration}")
        if task.image_bytes is None or task.js_code is None or task.rendered_png is None:
            await self._fail(task.stem, "critic: missing inputs", stage="critic")
            return
        artifact_context = {
            "kind": "coder_v1",
            "js_code": task.js_code,
            "osd": json.loads(task.osd) if task.osd else None,
        }
        try:
            report = await self.critic.critique(
                task_id=task.stem,
                image_bytes=task.image_bytes,
                image_mime=task.image_mime,
                render_png=task.rendered_png,
                artifact_context=artifact_context,
            )
        except Exception as exc:
            await self._fail(task.stem, f"critic: {type(exc).__name__}: {exc}", stage="critic")
            return
        await self.bus.publish(CriticDone(task_id=task.stem, report=report))

    async def _on_critic_done(self, event: Event) -> None:
        assert isinstance(event, CriticDone)
        task = self._tasks.get(event.task_id)
        if task is None or task.terminal:
            return
        report = event.report
  
        task.score_history.append(report.overall_score)
        if report.overall_score > task.best_score:
            task.best_score = report.overall_score
            task.best_iter = task.iteration
            task.best_js_code = task.js_code
            task.best_rendered_png = task.rendered_png
            logger.info(
                f"{self._stg(self._critic_stage, 'Critic')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration} | Score: {report.overall_score:.2f} | Best Updated"
            )
        # Happy path: critic is satisfied → complete.
        if report.stop or report.overall_score >= self.score_threshold or not report.issues:
            reason = (
                "stop_flag" if report.stop else
                "threshold_met" if report.overall_score >= self.score_threshold else
                "no_issues"
            )
            logger.info(
                f"{self._stg(self._critic_stage, 'Critic')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration} | Score: {report.overall_score:.2f} | Issues: {len(report.issues)} | Accepted: {reason}"
            )
            await self._complete(task.stem)
            return
        # Out of iterations: serve the best snapshot.
        if task.iteration >= self.max_iter:
            logger.info(
                f"{self._stg(self._critic_stage, 'Critic')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration} | Score: {report.overall_score:.2f} | Best: {task.best_score:.2f} | History: {task.score_history} | Max Iter Reached"
            )
            await self._complete(task.stem)
            return
        # F1 adaptive abort — give up early on hopeless tasks.
        if task.iteration >= 1 and task.best_score < 0.20:
            logger.info(
                f"{self._stg(self._critic_stage, 'Critic')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration} | Best: {task.best_score:.2f} | Adaptive Abort"
            )
            await self._complete(task.stem)
            return
        # Patch round.
        issue_kinds: dict[str, int] = {}
        for issue in report.issues:
            k = getattr(issue.kind, "value", str(issue.kind))
            issue_kinds[k] = issue_kinds.get(k, 0) + 1
        logger.info(
            f"{self._stg(self._patcher_stage, 'Patcher')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration} | Score: {report.overall_score:.2f} | Issues: {len(report.issues)} | By Kind: {issue_kinds}"
        )
        try:
            osd = OSD.model_validate_json(task.osd) if task.osd else None
            js_code = await self.coder.code_critic_repair(
                task_id=task.stem,
                osd=osd,
                issues=report.issues,
                overall_score=report.overall_score,
                matching_aspects=list(getattr(report, "matching_aspects", []) or []),
                image_bytes=task.image_bytes if self.coder_multimodal else None,
                image_mime=task.image_mime,
                render_png=task.rendered_png if self.coder_multimodal else None,
            )
            task.js_code = js_code
            task.iteration += 1
            applied_ops = self._issue_labels(report.issues)
            await self.bus.publish(PatcherDone(
                task_id=task.stem, js_code=js_code,
                applied_ops=applied_ops, iteration=task.iteration,
            ))
        except Exception as exc:
            await self._fail(task.stem, f"patcher: {type(exc).__name__}: {exc}", stage="patcher")

    async def _on_terminal(self, event: Event) -> None:
        task = self._tasks.get(event.task_id)
        if task is None:
            return
        task.terminal = True
        if task.deadline_task and not task.deadline_task.done():
            task.deadline_task.cancel()
        self.session_store.evict(event.task_id)
        self._tasks.pop(event.task_id, None)    
        future = self._futures.pop(event.task_id, None)
        if future is not None and not future.done():
            future.set_result(task)
        total_elapsed = time.time() - task.started_at
        if isinstance(event, TaskDone):
            logger.info(
                f"{self._stg(self._terminal_stage, 'Done')} Event: {event.type} | Task {task.stem} | Iter: {task.iteration} | Elapsed: {total_elapsed:.1f}s | Patches: {task.iteration}"
            )
        else:
            reason = getattr(event, "error", "?")
            stage = getattr(event, "stage", None)
            logger.warning(
                f"{self._stg(self._terminal_stage, 'Failed')} Event: {event.type} | Task {task.stem} | Reason: {reason} | Stage: {stage} | Elapsed: {total_elapsed:.1f}s"
            )

    # Helpers

    @staticmethod
    def _issue_labels(issues: list) -> list[str]:
        labels: list[str] = []
        for issue in issues:
            if isinstance(issue, dict):
                kind = issue.get("kind", "issue")
                desc = issue.get("description", "")
            else:
                kind = getattr(issue, "kind", "issue")
                if hasattr(kind, "value"):
                    kind = kind.value
                desc = getattr(issue, "description", "")
            labels.append(f"{kind}:{str(desc)[:80]}")
        return labels

    async def _complete(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task is None or task.terminal:
            return
        if task.best_js_code is not None:
            task.js_code = task.best_js_code
            task.rendered_png = task.best_rendered_png
            logger.info(
                f"{self._stg(self._terminal_stage, 'Done')} Event: task.done | Task {task.stem} | Source: best_iter | Best Iter: {task.best_iter} | Best Score: {task.best_score:.2f} | History: {task.score_history}"
            )
        else:
            logger.info(
                f"{self._stg(self._terminal_stage, 'Done')} Event: task.done | Task {task.stem} | Source: last_state"
            )
        artifact = {
            "js": task.js_code,
            "rendered_png": task.rendered_png,
        }
        task.failed = False
        task.failure_reason = None
        task.failure_stage = None
        await self.bus.publish(TaskDone(task_id=task_id, artifact=artifact))

    async def _fail(self, task_id: str, reason: str, *, stage: str | None = None) -> None:
        task = self._tasks.get(task_id)
        if task is None or task.terminal:
            return
        task.failed = True
        task.failure_reason = reason
        task.failure_stage = stage
        await self.bus.publish(TaskFailed(task_id=task_id, error=reason, stage=stage))

    async def _deadline_watcher(self, task_id: str) -> None:
        try:
            await asyncio.sleep(self.task_deadline_s)
            task = self._tasks.get(task_id)
            if task is not None and not task.terminal:
                await self._fail(task_id, "budget exceeded", stage="deadline")
        except asyncio.CancelledError:
            return
