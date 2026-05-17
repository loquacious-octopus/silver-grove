from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("PROBE_LOG_DIR", "/tmp/qwen35-probe-logs"))
APP_LOG = LOG_DIR / "batch_service.log"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-397B-A17B-FP8")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_ID)
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "local")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", f"http://127.0.0.1:{VLLM_PORT}/v1").rstrip("/")
NODE_BINARY = os.environ.get("NODE_BINARY", "node")
PROFILE_FILE = Path(os.environ.get("STAGE_PROFILE_FILE", ROOT / "profiles" / "stage_04_qwen35.md"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "4"))
MIN_ITERATIONS = int(os.environ.get("MIN_ITERATIONS", "2"))
PLATEAU_WINDOW = int(os.environ.get("PLATEAU_WINDOW", "3"))
VALIDATION_REPAIRS = int(os.environ.get("VALIDATION_REPAIRS", "2"))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.85"))
BATCH_CONCURRENCY = max(1, int(os.environ.get("BATCH_CONCURRENCY", "1")))
CHAT_TIMEOUT_S = float(os.environ.get("CHAT_TIMEOUT_S", "1200"))
RENDER_TIMEOUT_S = float(os.environ.get("RENDER_TIMEOUT_S", "90"))
CHECK_TIMEOUT_S = float(os.environ.get("CHECK_TIMEOUT_S", "20"))


def _log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with APP_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not stem:
        raise ValueError("prompt stem cannot be empty")
    return stem[:160]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("no JSON object found")


def _sanitize_js(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:javascript|js)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    marker = "export default function generate(THREE)"
    marker_index = raw.find(marker)
    if marker_index > 0:
        raw = raw[marker_index:]
    return raw.strip()


def _image_data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


def _profile_text() -> str:
    try:
        return PROFILE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "Stage 4 dense feature profile: prioritize primary body geometry, "
            "attached handles/supports, real negative space, repeated small "
            "features, and best-valid-candidate selection."
        )


class PromptItem(BaseModel):
    stem: str
    image_url: str


class GenerateRequest(BaseModel):
    prompts: list[PromptItem] = Field(min_length=1)
    seed: int = 42


class GenerateAccepted(BaseModel):
    accepted: int


class StatusResponse(BaseModel):
    status: str
    progress: int
    total: int
    payload: Any = None


class MinerStatus(str, Enum):
    WARMING_UP = "warming_up"
    READY = "ready"
    GENERATING = "generating"
    COMPLETE = "complete"
    REPLACE = "replace"


@dataclass
class IterationRecord:
    iteration: int
    js_code: str
    validation: dict[str, Any]
    review: dict[str, Any] = field(default_factory=dict)
    rendered_png: bytes | None = None


@dataclass
class PromptTask:
    stem: str
    image_url: str
    seed: int
    image_bytes: bytes | None = None
    image_mime: str = "image/jpeg"
    prompt_observation_md: str = ""
    initial_osd: dict[str, Any] = field(default_factory=dict)
    final_osd: dict[str, Any] = field(default_factory=dict)
    quality_gate: dict[str, Any] = field(default_factory=dict)
    lessons: dict[str, Any] = field(default_factory=dict)
    iteration_records: list[IterationRecord] = field(default_factory=list)
    js_code: str | None = None
    js_valid: bool = False
    js_errors: list[str] = field(default_factory=list)
    js_metrics: dict[str, Any] = field(default_factory=dict)
    rendered_png: bytes | None = None
    render_errors: list[str] = field(default_factory=list)
    best_score: float | None = None
    best_iter: int | None = None
    failed: bool = False
    failure_reason: str = ""
    started_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None


class BatchState:
    def __init__(self) -> None:
        self.status = MinerStatus.WARMING_UP
        self.total = 0
        self.seed = 0
        self.batch_stems: list[str] = []
        self.tasks: dict[str, PromptTask] = {}
        self.failed: dict[str, str] = {}
        self.started_at: str | None = None
        self.completed_at: str | None = None

    @property
    def progress(self) -> int:
        return len(self.tasks)

    def reset_for_batch(self, stems: list[str], seed: int) -> None:
        self.status = MinerStatus.GENERATING
        self.total = len(stems)
        self.seed = seed
        self.batch_stems = stems
        self.tasks = {}
        self.failed = {}
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = None

    def record_task(self, task: PromptTask) -> None:
        self.tasks[task.stem] = task
        if task.failed:
            self.failed[task.stem] = task.failure_reason or "failed"

    def mark_complete(self) -> None:
        self.status = MinerStatus.COMPLETE
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_response(self) -> dict[str, Any]:
        active = self.status in {MinerStatus.GENERATING, MinerStatus.COMPLETE}
        return {
            "status": self.status.value,
            "progress": self.progress if active else 0,
            "total": self.total if active else 0,
            "payload": {
                "status_upper": self.status.value.upper(),
                "seed": self.seed,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "failed": self.failed,
                "model": SERVED_MODEL_NAME,
                "local_vllm_base_url": VLLM_BASE_URL,
            },
        }


class LocalVLLMClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(CHAT_TIMEOUT_S, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def ready(self) -> tuple[bool, Any]:
        try:
            response = await self._client.get(
                f"{VLLM_BASE_URL}/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=10.0,
            )
            body = response.json()
            return response.status_code == 200 and SERVED_MODEL_NAME in json.dumps(body), body
        except Exception as exc:  # noqa: BLE001
            return False, {"error": repr(exc)}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        retries: int = 2,
    ) -> str:
        payload: dict[str, Any] = {
            "model": SERVED_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = await self._client.post(
                    f"{VLLM_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"].get("content") or ""
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content if isinstance(part, dict))
                return str(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if response_format is not None and "response_format" in str(exc):
                    payload.pop("response_format", None)
                    response_format = None
                    continue
                await asyncio.sleep(min(5.0, attempt))
        if last_error is None:
            raise RuntimeError("local vLLM chat failed: no response")
        raise RuntimeError(f"local vLLM chat failed: {type(last_error).__name__}: {last_error!r}")


class ModelingEngine:
    def __init__(self, client: LocalVLLMClient) -> None:
        self.client = client
        self.profile = _profile_text()

    async def run_batch(self, prompts: list[PromptItem], seed: int, state: BatchState) -> None:
        sem = asyncio.Semaphore(BATCH_CONCURRENCY)

        async def run_one(item: PromptItem) -> None:
            async with sem:
                task = PromptTask(stem=_safe_stem(item.stem), image_url=item.image_url, seed=seed)
                try:
                    await self.run_task(task)
                except Exception as exc:  # noqa: BLE001
                    task.failed = True
                    task.failure_reason = f"{type(exc).__name__}: {exc}"
                    _log(f"[{task.stem}] failed: {task.failure_reason}")
                finally:
                    task.completed_at = time.monotonic()
                    state.record_task(task)
                    _log(f"[batch] progress {state.progress}/{state.total}")

        await asyncio.gather(*(run_one(prompt) for prompt in prompts))
        state.mark_complete()

    async def run_task(self, task: PromptTask) -> None:
        _log(f"[{task.stem}] fetch image")
        task.image_bytes, task.image_mime = await self._fetch_image(task.image_url)
        observation = await self._observe(task)
        task.prompt_observation_md = observation.get("prompt_observation_md", "")
        task.initial_osd = observation.get("initial_osd", {})
        previous_js: str | None = None
        critic_report: dict[str, Any] | None = None
        champion: tuple[float, int, str, dict[str, Any], bytes | None, dict[str, Any]] | None = None

        for iteration in range(1, MAX_ITERATIONS + 1):
            _log(f"[{task.stem}] iteration {iteration}")
            if iteration == 1:
                js_code = await self._code_fresh(task)
            else:
                js_code = await self._code_repair(task, previous_js or "", critic_report or {})

            validation = await self._validate_with_repairs(task, js_code)
            js_code = validation.pop("_js_code")
            rendered = None
            render_errors: list[str] = []
            review: dict[str, Any] = {}
            if validation.get("passed"):
                rendered, render_errors = await self._render(js_code)
                if rendered:
                    review = await self._critic(task, rendered, champion[4] if champion else None, iteration)
                    score = review.get("overall_score")
                    if isinstance(score, (int, float)):
                        if champion is None or float(score) >= champion[0]:
                            champion = (float(score), iteration, js_code, validation, rendered, review)
                    elif champion is None:
                        review["critic_status"] = "numeric_score_missing"
                        champion = (-1.0, iteration, js_code, validation, rendered, review)
                elif champion is None:
                    review = {"critic_status": "render_failed", "render_errors": render_errors}
                    champion = (-1.0, iteration, js_code, validation, None, review)
            else:
                review = {"critic_status": "validation_failed", "issues": validation.get("failures", [])}

            task.iteration_records.append(
                IterationRecord(
                    iteration=iteration,
                    js_code=js_code,
                    validation=validation,
                    review=review,
                    rendered_png=rendered,
                )
            )
            previous_js = js_code
            critic_report = review
            score = review.get("overall_score")
            if iteration >= MIN_ITERATIONS:
                if isinstance(score, (int, float)) and float(score) >= SCORE_THRESHOLD:
                    break
                if champion is not None and iteration - champion[1] >= PLATEAU_WINDOW:
                    break

        if champion is None:
            task.failed = True
            task.failure_reason = "no valid JavaScript candidate produced"
            return

        best_score, best_iter, best_js, best_validation, best_png, best_review = champion
        task.js_code = best_js.strip() + "\n"
        task.js_valid = bool(best_validation.get("passed"))
        task.js_errors = [
            f"{item.get('rule')}: {item.get('detail')}" for item in best_validation.get("failures", [])
        ]
        task.js_metrics = dict(best_validation.get("metrics") or {})
        task.rendered_png = best_png
        task.best_score = best_score if best_score >= 0 else None
        task.best_iter = best_iter
        task.final_osd = {
            **task.initial_osd,
            "best_iteration": best_iter,
            "best_score": task.best_score,
            "iterations": len(task.iteration_records),
        }
        task.quality_gate = {
            "status": "passed_quality_gate" if task.best_score is not None and task.best_score >= SCORE_THRESHOLD else "best_valid_candidate",
            "best_score": task.best_score,
            "best_iter": best_iter,
            "score_threshold": SCORE_THRESHOLD,
            "min_iterations": MIN_ITERATIONS,
            "max_iterations": MAX_ITERATIONS,
            "plateau_window": PLATEAU_WINDOW,
            "score_history": [
                rec.review.get("overall_score") for rec in task.iteration_records if isinstance(rec.review.get("overall_score"), (int, float))
            ],
            "critic_status": best_review.get("critic_status", "scored"),
            "local_vllm_model": SERVED_MODEL_NAME,
        }
        task.lessons = {
            "selected_repair_kind": best_review.get("selected_repair_kind"),
            "selected_repair_target_node_id": best_review.get("selected_repair_target_node_id"),
            "biggest_flaw": best_review.get("biggest_flaw") or best_review.get("selected_repair"),
        }

    async def _fetch_image(self, url: str) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            mime = response.headers.get("content-type", "image/jpeg").split(";")[0]
            return response.content, mime or "image/jpeg"

    async def _observe(self, task: PromptTask) -> dict[str, Any]:
        prompt = (
            "Inspect the image and return JSON only. Build a feature ledger for a Three.js reconstruction.\n"
            "Required keys: prompt_observation_md, initial_osd.\n"
            "initial_osd should include primary_object_type, coordinate_frame, major_volumes, small_features, "
            "repeated_elements, materials, contact_points, orientation_invariants, primitive_plan, part_graph.\n\n"
            f"Stage profile:\n{self.profile}"
        )
        try:
            text = await self.client.chat(
                [
                    {"role": "system", "content": "You are a precise visual feature ledger writer. Return JSON only."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _image_data_url(task.image_bytes or b"", task.image_mime)}},
                        ],
                    },
                ],
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            payload = _extract_json_object(text)
            if isinstance(payload.get("initial_osd"), dict):
                return payload
        except Exception as exc:  # noqa: BLE001
            _log(f"[{task.stem}] observe fallback: {exc}")
        object_name = task.stem.replace("_", " ")
        return {
            "prompt_observation_md": f"# Feature Ledger\n- Primary object: {object_name}\n- Reconstruct from prompt image.",
            "initial_osd": {
                "primary_object_type": object_name,
                "coordinate_frame": "infer from prompt image",
                "major_volumes": [object_name],
                "small_features": [],
                "repeated_elements": [],
                "materials": [],
                "contact_points": [],
                "orientation_invariants": ["preserve prompt-facing orientation"],
                "primitive_plan": ["choose literal primitives from image"],
                "part_graph": [object_name],
            },
        }

    async def _code_fresh(self, task: PromptTask) -> str:
        prompt = self._coder_prompt(task, previous_js="", critic_report={}, phase="fresh")
        text = await self.client.chat(
            [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=6000,
        )
        return _sanitize_js(text)

    async def _code_repair(self, task: PromptTask, previous_js: str, critic_report: dict[str, Any]) -> str:
        prompt = self._coder_prompt(task, previous_js=previous_js, critic_report=critic_report, phase="critic_repair")
        text = await self.client.chat(
            [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=6000,
        )
        return _sanitize_js(text)

    def _coder_prompt(self, task: PromptTask, previous_js: str, critic_report: dict[str, Any], phase: str) -> str:
        parts = [
            "Prompt observation ledger:",
            task.prompt_observation_md,
            "",
            "Initial OSD JSON:",
            json.dumps(task.initial_osd, indent=2),
            "",
            "Stage profile:",
            self.profile,
            "",
            "Output requirements:",
            "- Return only JavaScript code.",
            "- The first non-whitespace characters must be: export default function generate(THREE)",
            "- Do not create a scene, camera, renderer, animation loop, DOM, or browser globals.",
            "- Return a THREE.Group or Object3D centered near the origin and fitting inside a unit-scale box.",
            "- Use named groups/variables for major parts so repairs can target them.",
            "- Make handles, rods, legs, brackets, wheels, and supports visibly intersect or attach to parents.",
            "- Use real gaps/separate geometry for holes, slots, filigree, grilles, spokes, and openings.",
            "- Use repeated primitives for repeated details; do not describe them as texture only.",
        ]
        if phase == "fresh":
            parts.extend(["", "Generate the full module from scratch now."])
        else:
            parts.extend(
                [
                    "",
                    "Previous JavaScript module:",
                    previous_js,
                    "",
                    "Critic or validation issue to repair:",
                    json.dumps(critic_report, indent=2),
                    "",
                    "Repair exactly the highest-impact issue while preserving working object identity.",
                ]
            )
        return "\n".join(parts)

    async def _validate_with_repairs(self, task: PromptTask, js_code: str) -> dict[str, Any]:
        current = js_code
        last_result: dict[str, Any] = {}
        for attempt in range(VALIDATION_REPAIRS + 1):
            result = await run_js_validator(current)
            if result.get("passed"):
                result["_js_code"] = current
                return result
            last_result = result
            if attempt >= VALIDATION_REPAIRS:
                break
            repair_prompt = self._coder_prompt(
                task,
                previous_js=current,
                critic_report={"validation_failures": result.get("failures", [])},
                phase="validation_repair",
            )
            text = await self.client.chat(
                [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ],
                max_tokens=6000,
            )
            current = _sanitize_js(text)
        last_result["_js_code"] = current
        return last_result

    async def _render(self, js_code: str) -> tuple[bytes | None, list[str]]:
        return await run_renderer(js_code)

    async def _critic(
        self,
        task: PromptTask,
        rendered_png: bytes,
        champion_png: bytes | None,
        iteration: int,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Compare the original prompt image to the candidate 2x2 render grid. "
                    "Return JSON only with keys: overall_score (0-1), matching_aspects, issues, "
                    "selected_repair_kind, selected_repair_target_node_id, accept_or_reject, biggest_flaw. "
                    "Do not omit overall_score."
                ),
            },
            {"type": "image_url", "image_url": {"url": _image_data_url(task.image_bytes or b"", task.image_mime)}},
            {"type": "image_url", "image_url": {"url": _image_data_url(rendered_png, "image/png")}},
        ]
        if champion_png is not None:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(champion_png, "image/png")}})
        try:
            text = await self.client.chat(
                [
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=1800,
                response_format={"type": "json_object"},
            )
            payload = _extract_json_object(text)
            score = payload.get("overall_score", payload.get("visual_score"))
            if not isinstance(score, (int, float)):
                recovery = await self.client.chat(
                    [
                        {"role": "system", "content": "Return JSON only with one numeric key overall_score from 0.0 to 1.0."},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                recovered = _extract_json_object(recovery)
                score = recovered.get("overall_score")
                if isinstance(score, (int, float)):
                    payload["overall_score"] = float(score)
                    payload["critic_status"] = "score_only_recovery"
            else:
                payload["overall_score"] = float(score)
            payload["iteration"] = iteration
            return payload
        except Exception as exc:  # noqa: BLE001
            return {"iteration": iteration, "critic_status": "critic_failed", "critic_error": f"{type(exc).__name__}: {exc}"}


CODER_SYSTEM_PROMPT = (
    "You generate procedural Three.js object modules for visual reconstruction. "
    "Return only complete JavaScript. The module must default-export "
    "`function generate(THREE)` and return a THREE.Object3D. Avoid prose."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a strict visual critic for procedural 3D object reconstruction. "
    "Inspect the original image and rendered grid as a multi-view object. "
    "Score object identity, silhouette, orientation, attachments, negative space, "
    "repeated features, and materials. Return JSON only."
)


async def run_js_validator(js_code: str) -> dict[str, Any]:
    if shutil.which(NODE_BINARY) is None:
        return {"passed": False, "failures": [{"rule": "NODE_MISSING", "detail": NODE_BINARY}]}
    runner = ROOT / "js_validate.mjs"
    with tempfile.TemporaryDirectory(prefix="steady_validate_") as tmp:
        path = Path(tmp) / "candidate.mjs"
        path.write_text(js_code, encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            NODE_BINARY,
            str(runner),
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=CHECK_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"passed": False, "failures": [{"rule": "VALIDATION_TIMEOUT", "detail": "node validator timed out"}]}
        if proc.returncode != 0:
            return {
                "passed": False,
                "failures": [{"rule": "VALIDATOR_FAILED", "detail": stderr.decode(errors="replace")[-1000:]}],
            }
        try:
            return json.loads(stdout.decode())
        except Exception as exc:  # noqa: BLE001
            return {"passed": False, "failures": [{"rule": "VALIDATOR_JSON_FAILED", "detail": repr(exc)}]}


async def run_renderer(js_code: str) -> tuple[bytes | None, list[str]]:
    if shutil.which(NODE_BINARY) is None:
        return None, [f"node missing: {NODE_BINARY}"]
    runner = ROOT / "render_grid.mjs"
    with tempfile.TemporaryDirectory(prefix="steady_render_") as tmp:
        module_path = Path(tmp) / "candidate.mjs"
        output_path = Path(tmp) / "grid.png"
        module_path.write_text(js_code, encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            NODE_BINARY,
            str(runner),
            str(module_path),
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT),
        )
        try:
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=RENDER_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None, ["renderer timed out"]
        if proc.returncode != 0 or not output_path.exists():
            return None, [stderr.decode(errors="replace")[-2000:] or f"renderer exited {proc.returncode}"]
        return output_path.read_bytes(), []


state = BatchState()
client = LocalVLLMClient()
engine = ModelingEngine(client)
_generate_lock = asyncio.Lock()
_generation_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ready, body = await client.ready()
    state.status = MinerStatus.READY if ready else MinerStatus.WARMING_UP
    _log(f"batch service starting | ready={ready} models={str(body)[:500]}")
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="Steady Harbor Qwen3.5 Batch Service", lifespan=lifespan)


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.get("/ready")
async def ready() -> JSONResponse:
    ok, body = await client.ready()
    if ok and state.status == MinerStatus.WARMING_UP:
        state.status = MinerStatus.READY
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "models": body})


@app.get("/status", response_model=StatusResponse)
async def status(replacements_remaining: int = 0) -> dict[str, Any]:
    del replacements_remaining
    if state.status == MinerStatus.WARMING_UP:
        ok, _ = await client.ready()
        if ok:
            state.status = MinerStatus.READY
    return state.to_response()


@app.post("/generate", response_model=GenerateAccepted)
async def generate(request: GenerateRequest) -> GenerateAccepted:
    global _generation_task
    async with _generate_lock:
        ok, _ = await client.ready()
        if not ok:
            state.status = MinerStatus.WARMING_UP
            raise HTTPException(503, "local vLLM is not ready")

        stems = [_safe_stem(prompt.stem) for prompt in request.prompts]
        if state.status == MinerStatus.GENERATING and sorted(state.batch_stems) == sorted(stems):
            return GenerateAccepted(accepted=len(request.prompts))
        if state.status not in {MinerStatus.READY, MinerStatus.COMPLETE, MinerStatus.WARMING_UP}:
            raise HTTPException(409, f"Cannot accept batch while status={state.status.value}")

        if _generation_task and not _generation_task.done():
            _generation_task.cancel()
            try:
                await _generation_task
            except asyncio.CancelledError:
                pass

        normalized = [PromptItem(stem=_safe_stem(p.stem), image_url=p.image_url) for p in request.prompts]
        state.reset_for_batch([p.stem for p in normalized], request.seed)

        async def _run_safe() -> None:
            try:
                await engine.run_batch(normalized, request.seed, state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _log(f"batch crashed: {type(exc).__name__}: {exc}")
                state.mark_complete()

        _generation_task = asyncio.create_task(_run_safe())
    return GenerateAccepted(accepted=len(request.prompts))


@app.get("/results")
async def results() -> StreamingResponse:
    if state.status != MinerStatus.COMPLETE:
        raise HTTPException(409, f"Not complete, current status: {state.status.value}")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem in state.batch_stems:
            task = state.tasks.get(stem)
            if task is None or task.failed or not task.js_code:
                continue
            zf.writestr(f"{stem}.js", task.js_code)
            base = f"prompts/{stem}"
            zf.writestr(f"{base}/prompt_observation.md", task.prompt_observation_md or "")
            zf.writestr(f"{base}/initial_osd.json", json.dumps(task.initial_osd or {}, indent=2))
            zf.writestr(f"{base}/final_osd.json", json.dumps(task.final_osd or {}, indent=2))
            zf.writestr(f"{base}/quality_gate.json", json.dumps(task.quality_gate or {}, indent=2))
            zf.writestr(f"{base}/lessons.json", json.dumps(task.lessons or {}, indent=2))
            zf.writestr(f"{base}/final_candidate.js", task.js_code)
            zf.writestr(
                f"{base}/final_candidate.validation.json",
                json.dumps(
                    {
                        "passed": task.js_valid,
                        "failures": task.js_errors,
                        "metrics": task.js_metrics,
                    },
                    indent=2,
                ),
            )
            if task.rendered_png:
                zf.writestr(f"{base}/final_render.png", task.rendered_png)
            for record in task.iteration_records:
                prefix = f"{base}/iterations/iter_{record.iteration:03d}"
                zf.writestr(f"{prefix}.js", record.js_code)
                zf.writestr(f"{prefix}.validation.json", json.dumps(record.validation, indent=2))
                zf.writestr(f"{prefix}_review.json", json.dumps(record.review, indent=2))
                if record.rendered_png:
                    zf.writestr(f"{prefix}.grid.png", record.rendered_png)
        if state.failed:
            zf.writestr(
                "_failed.json",
                json.dumps([{"stem": stem, "reason": reason} for stem, reason in state.failed.items()], indent=2),
            )
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip")


@app.get("/debug/tasks")
async def debug_tasks() -> dict[str, Any]:
    tasks = []
    for stem in state.batch_stems:
        task = state.tasks.get(stem)
        if task is None:
            tasks.append({"stem": stem, "status": "in_progress"})
            continue
        tasks.append(
            {
                "stem": stem,
                "status": "failed" if task.failed else "completed",
                "failed": task.failed,
                "failure_reason": task.failure_reason,
                "best_score": task.best_score,
                "best_iter": task.best_iter,
                "js_valid": task.js_valid,
                "js_code_bytes": len(task.js_code.encode("utf-8")) if task.js_code else 0,
                "has_png": task.rendered_png is not None,
                "iterations": len(task.iteration_records),
            }
        )
    return {"status": state.status.value, "progress": state.progress, "total": state.total, "tasks": tasks}


@app.get("/debug/tasks/{stem}")
async def debug_task(stem: str) -> dict[str, Any]:
    safe = _safe_stem(stem)
    task = state.tasks.get(safe)
    if task is None:
        raise HTTPException(404, "task not found")
    return {
        "stem": task.stem,
        "image_url": task.image_url,
        "failed": task.failed,
        "failure_reason": task.failure_reason,
        "js_valid": task.js_valid,
        "js_errors": task.js_errors,
        "js_metrics": task.js_metrics,
        "best_score": task.best_score,
        "best_iter": task.best_iter,
        "quality_gate": task.quality_gate,
        "rendered_png_b64": base64.b64encode(task.rendered_png).decode() if task.rendered_png else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("APP_PORT", "10006")))
