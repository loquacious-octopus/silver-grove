from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from config.settings import SettingsConf
from llm.session_store import SessionStore
from modules.critic.agent import CriticAgent
from modules.js_checker.module import JSCheckerModule
from modules.judge.agent import JudgeAgent
from modules.renderer.module import RendererModule
from modules.scene_coder.agent import SceneCoderAgent
from modules.scene_planner.agent import ScenePlannerAgent
from pipeline.bus import EventBus
from pipeline.orchestrator import Orchestrator
from pipeline.task import PipelineTask


@dataclass
class Pipeline:
    """Bundle of the full graph. Hold onto this across app lifecycle."""

    orchestrator: Orchestrator
    bus: EventBus
    session_store: SessionStore
    planner: ScenePlannerAgent | None
    coder: SceneCoderAgent

    async def start(self) -> None:
        await self.orchestrator.start()

    async def stop(self) -> None:
        await self.orchestrator.stop()

    async def submit(self, task: PipelineTask) -> asyncio.Future[PipelineTask]:
        return await self.orchestrator.submit(task)


def build_pipeline(
    *,
    settings: SettingsConf,
    clients: dict[str, Any],
    js_checker: JSCheckerModule,
    renderer: RendererModule,
    http_client: httpx.AsyncClient,
    session_store: SessionStore | None = None,
) -> Pipeline:
    """Wire agents + orchestrator from settings."""
    session_store = session_store or SessionStore()
    actors = settings.actors
    policy = settings.event_bus
    llm = settings.llm_clients
    use_planner = settings.pipeline.use_planner
    total_stages = 7 if use_planner else 6

    def _backend(actor_client: str) -> str:
        cfg = llm.get(actor_client)
        return cfg.backend if cfg is not None else "openrouter"

    if use_planner:
        planner: ScenePlannerAgent | None = ScenePlannerAgent(
            client=clients[actors.planner.client],
            model=actors.planner.model,
            session_store=session_store,
            temperature=actors.planner.temperature,
            seed=actors.planner.seed,
            max_tokens=actors.planner.max_tokens,
            reasoning_effort=actors.planner.reasoning_effort,
            backend=_backend(actors.planner.client),
        )
    else:
        planner = None

    coder = SceneCoderAgent(
        client=clients[actors.coder.client],
        model=actors.coder.model,
        session_store=session_store,
        temperature=actors.coder.temperature,
        seed=actors.coder.seed,
        max_tokens=actors.coder.max_tokens,
        backend=_backend(actors.coder.client),
        total_stages=total_stages,
    )

    critic = CriticAgent(
        client=clients[actors.critic.client],
        model=actors.critic.model,
        max_tokens=actors.critic.max_tokens,
        seed=actors.critic.seed,
        reasoning_effort=actors.critic.reasoning_effort,
        ensemble_size=actors.critic.ensemble_size,
        backend=_backend(actors.critic.client),
        total_stages=total_stages,
    )

    if actors.coder.ensemble_size > 1:
        judge: JudgeAgent | None = JudgeAgent(
            client=clients[actors.judge.client],
            model=actors.judge.model,
            max_tokens=actors.judge.max_tokens,
            seed=actors.judge.seed,
            reasoning_effort=actors.judge.reasoning_effort,
            backend=_backend(actors.judge.client),
        )
    else:
        judge = None

    bus = EventBus()
    queue_sizes = [
        actors.coder.queue_size,
        actors.checker.queue_size,
        actors.renderer.queue_size,
        actors.critic.queue_size,
        actors.patcher.queue_size,
    ]
    if use_planner:
        queue_sizes.append(actors.planner.queue_size)
    orchestrator = Orchestrator(
        bus=bus,
        session_store=session_store,
        planner=planner,
        coder=coder,
        critic=critic,
        judge=judge,
        js_checker=js_checker,
        renderer=renderer,
        http_client=http_client,
        coder_multimodal=actors.coder.multimodal,
        task_deadline_s=policy.task_deadline_s,
        max_iter=policy.max_iter,
        score_threshold=policy.score_threshold,
        planner_workers=actors.planner.workers if use_planner else 0,
        coder_workers=actors.coder.workers,
        checker_workers=actors.checker.workers,
        renderer_workers=actors.renderer.workers,
        critic_workers=actors.critic.workers,
        patcher_workers=actors.patcher.workers,
        queue_size=max(queue_sizes),
        coder_ensemble_size=actors.coder.ensemble_size,
        coder_ensemble_temperature=actors.coder.ensemble_temperature,
    )

    return Pipeline(
        orchestrator=orchestrator,
        bus=bus,
        session_store=session_store,
        planner=planner,
        coder=coder,
    )
