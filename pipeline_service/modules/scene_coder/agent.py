from __future__ import annotations

import base64
import json
import time
from typing import Any

from llm.session import SessionAgent
from llm.session_store import SessionStore
from logger_config import logger
from modules.critic.schema import Issue
from modules.scene_coder.prompts import (
    CODER_SYSTEM_PROMPT,
    CODER_USER_TEMPLATE_CHECKER_REPAIR,
    CODER_USER_TEMPLATE_CHECKER_REPAIR_IMAGE,
    CODER_USER_TEMPLATE_CRITIC_REPAIR,
    CODER_USER_TEMPLATE_CRITIC_REPAIR_IMAGE,
    CODER_USER_TEMPLATE_FRESH,
    CODER_USER_TEMPLATE_OSD,
)
from modules.scene_planner.schema import OSD
from modules.scene_coder.portfolio_strategy import candidate_user_suffix

_ACTOR = "coder"


class SceneCoderAgent:
    """Per-pipeline JS code generator."""

    actor = _ACTOR

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        session_store: SessionStore,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int = 8192,
        max_tool_iters: int = 4,
        max_output_retries: int = 2,
        backend: str = "openrouter",
        total_stages: int = 7,
    ) -> None:
        self.client = client
        self.model = model
        self.session_store = session_store
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_tool_iters = max_tool_iters
        self.max_output_retries = max_output_retries
        self.backend = backend
        self.total_stages = total_stages
        self._coder_stage = 2 if total_stages == 7 else 1
        self._patcher_stage = 6 if total_stages == 7 else 5

    def _build_session(self, task_id: str, actor: str) -> SessionAgent:
        return self._build_session_with(
            task_id, actor, seed=self.seed, temperature=self.temperature,
        )

    def _build_session_with(
        self, task_id: str, actor: str, *, seed: int | None, temperature: float,
    ) -> SessionAgent:
        return SessionAgent(
            task_id=task_id,
            actor=actor,
            system_prompt=CODER_SYSTEM_PROMPT,
            model=self.model,
            tools=None,
            response_model=None,
            client=self.client,
            temperature=temperature,
            seed=seed,
            max_tokens=self.max_tokens,
            max_tool_iters=self.max_tool_iters,
            backend=self.backend,
        )

    async def code(
        self,
        task_id: str,
        *,
        osd: OSD | None = None,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        image_mime: str = "image/jpeg",
        candidate_id: int = 0,
        seed_override: int | None = None,
        temperature_override: float | None = None,
    ) -> str:
        if osd is None and image_bytes is None and image_url is None:
            raise ValueError(
                "code() requires osd OR image_bytes/image_url "
                "(image-direct mode is used when planner is disabled)."
            )
        actor = f"{self.actor}#k{candidate_id}" if candidate_id > 0 else self.actor
        seed = seed_override if seed_override is not None else self.seed
        temperature = (
            temperature_override if temperature_override is not None
            else self.temperature
        )
        factory = (
            self._build_session
            if (seed_override is None and temperature_override is None)
            else (lambda tid, a: self._build_session_with(
                tid, a, seed=seed, temperature=temperature,
            ))
        )
        session = self.session_store.get_or_create(task_id, actor, factory)
        if osd is not None:
            text = CODER_USER_TEMPLATE_OSD.format(
                osd_json=osd.model_dump_json(indent=2)
            )
        else:
            text = CODER_USER_TEMPLATE_FRESH

        candidate_extra = candidate_user_suffix(candidate_id)
        if candidate_extra:
            text = text + "\n\n" + candidate_extra

        if image_bytes:
            ref_b64 = base64.b64encode(image_bytes).decode()
            url_str = f"data:{image_mime};base64,{ref_b64}"
            user_msg: str | list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": url_str}},
                {"type": "text", "text": text},
            ]
        elif image_url:
            user_msg = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ]
        else:
            user_msg = text
        prefix = f"[{self._coder_stage}/{self.total_stages} Coder]"
        parts_n = len(osd.parts) if osd is not None else 0
        cand_tag = f" | k{candidate_id} (seed={seed}, T={temperature})" if candidate_id > 0 else ""
        logger.info(
            f"{prefix} Started Task {task_id} | Model: {self.model} | OSD Parts: {parts_n} | Multimodal: {bool(image_bytes or image_url)}{cand_tag}"
        )
        t0 = time.time()
        js_code = await self._run_until_valid_js(session, user_msg)
        dt = time.time() - t0
        logger.info(
            f"{prefix} Finished Task {task_id} | Elapsed: {dt:.1f}s | Bytes: {len(js_code.encode('utf-8'))} | Lines: {len(js_code.splitlines())}{cand_tag}"
        )
        return js_code

    async def code_repair(
        self,
        task_id: str,
        *,
        osd: OSD | None = None,
        js_errors: list[str],
    ) -> str:
        session = self.session_store.get(task_id, self.actor)
        if session is None:
            raise RuntimeError(
                f"Code repair called for Task ID: {task_id} but no Coder session exists; "
                "code() must run first."
            )
        errors_block = "\n".join(f"- {e}" for e in js_errors) or "- (no specific errors returned)"
        if osd is not None:
            user_msg = CODER_USER_TEMPLATE_CHECKER_REPAIR.format(
                osd_json=osd.model_dump_json(indent=2),
                errors_block=errors_block,
            )
        else:
            user_msg = CODER_USER_TEMPLATE_CHECKER_REPAIR_IMAGE.format(
                errors_block=errors_block,
            )
        logger.info(
            f"[Coder Repair] Started Task {task_id} | Repair: checker | Errors: {len(js_errors)}"
        )
        t0 = time.time()
        js_code = await self._run_until_valid_js(session, user_msg)
        dt = time.time() - t0
        logger.info(
            f"[Coder Repair] Finished Task {task_id} | Elapsed: {dt:.1f}s | Bytes: {len(js_code.encode('utf-8'))}"
        )
        return js_code

    async def code_critic_repair(
        self,
        task_id: str,
        *,
        osd: OSD | None = None,
        issues: list[Issue] | list[dict[str, Any]],
        overall_score: float,
        matching_aspects: list[str] | None = None,
        image_bytes: bytes | None = None,
        image_mime: str = "image/png",
        render_png: bytes | None = None,
    ) -> str:
        """Run the Coder in repair mode."""
        session = self.session_store.get(task_id, self.actor)
        if session is None:
            raise RuntimeError(
                f"code_critic_repair called for Task ID: {task_id} but no Coder session exists; "
                "code() must run first."
            )
        normalized_issues = []
        for issue in issues:
            if hasattr(issue, "model_dump"):
                normalized_issues.append(issue.model_dump(mode="json"))
            else:
                normalized_issues.append(issue)
        matching_block = (
            "\n".join(f"- {m}" for m in matching_aspects)
            if matching_aspects else "- (none flagged by critic — proceed carefully)"
        )
        issues_json = json.dumps(normalized_issues, indent=2, ensure_ascii=False)
        if osd is not None:
            user_text = CODER_USER_TEMPLATE_CRITIC_REPAIR.format(
                osd_json=osd.model_dump_json(indent=2),
                overall_score=f"{overall_score:.2f}",
                issues_json=issues_json,
                matching_block=matching_block,
            )
        else:
            user_text = CODER_USER_TEMPLATE_CRITIC_REPAIR_IMAGE.format(
                overall_score=f"{overall_score:.2f}",
                issues_json=issues_json,
                matching_block=matching_block,
            )
        multimodal = image_bytes is not None and render_png is not None
        user_content: str | list[dict[str, Any]]
        if multimodal:
            ref_b64 = base64.b64encode(image_bytes).decode()
            render_b64 = base64.b64encode(render_png).decode()
            user_content = [
                {"type": "text",
                 "text": "Below: (1) the REFERENCE image we're trying to match, "
                         "and (2) the RENDER of your previous JS module. "
                         "Compare them yourself and decide what to fix. "
                         "Critic feedback follows but you should rely on the "
                         "visual comparison primarily.\n"},
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_mime};base64,{ref_b64}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{render_b64}"}},
                {"type": "text", "text": user_text},
            ]
        else:
            user_content = user_text

        patch_prefix = f"[{self._patcher_stage}/{self.total_stages} Patcher]"
        logger.info(
            f"{patch_prefix} Started Task {task_id} | Repair: critic | Issues: {len(normalized_issues)} | Score: {overall_score:.2f} | Multimodal: {multimodal}"
        )
        t0 = time.time()
        js_code = await self._run_until_valid_js(session, user_content)
        dt = time.time() - t0
        logger.info(
            f"{patch_prefix} Finished Task {task_id} | Elapsed: {dt:.1f}s | Bytes: {len(js_code.encode('utf-8'))}"
        )
        return js_code

    async def _run_until_valid_js(
        self,
        session: SessionAgent,
        user_msg: str | list[dict[str, Any]],
    ) -> str:
        raw = await session.run(user_msg)
        js = self._normalize_js_output(str(raw))
        for attempt in range(self.max_output_retries + 1):
            if self._looks_like_js_module(js):
                return js
            if attempt >= self.max_output_retries:
                break
            raw = await session.run(
                "Your previous response was not a valid raw JavaScript module. "
                "Return ONLY the full JS source with the exact signature "
                "`export default function generate(THREE)` and no markdown fences."
            )
            js = self._normalize_js_output(str(raw))
        raise ValueError(
            "Coder did not return a valid JS module with "
            "`export default function generate(THREE)`."
        )

    @staticmethod
    def _normalize_js_output(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _looks_like_js_module(text: str) -> bool:
        return (
            "export default function generate(THREE)" in text
            and "return" in text
        )
