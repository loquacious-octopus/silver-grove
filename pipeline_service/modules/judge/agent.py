from __future__ import annotations

import base64
import time
from typing import Any

from logger_config import logger
from modules.judge.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from modules.judge.schema import JudgeVerdict
from utils.json_extract import extract_json_object
from utils.retry import async_retry


class JudgeAgent:
    """Stateless pairwise judge. One call ↔ one JudgeVerdict."""

    actor = "judge"

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 1024,
        seed: int | None = 42,
        max_retries: int = 2,
        reasoning_effort: str | None = None,
        backend: str = "openrouter",
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.backend = backend

    async def compare(
        self,
        *,
        task_id: str,
        match_label: str,
        reference_bytes: bytes,
        reference_mime: str,
        render_a: bytes,
        render_b: bytes,
    ) -> JudgeVerdict:
        ref_b64 = base64.b64encode(reference_bytes).decode()
        a_b64 = base64.b64encode(render_a).decode()
        b_b64 = base64.b64encode(render_b).decode()

        extra_body: dict[str, Any] = {}
        if self.reasoning_effort:
            if self.backend == "vllm":
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                extra_body["reasoning"] = {"effort": self.reasoning_effort}

        prefix = f"[Judge {match_label}]"
        logger.info(
            f"{prefix} Started Task {task_id} | Model: {self.model} | "
            f"Ref KB: {len(reference_bytes) / 1024:.1f} | "
            f"A KB: {len(render_a) / 1024:.1f} | B KB: {len(render_b) / 1024:.1f}"
        )

        async def _call(_attempt: int, _last_err: str | None) -> JudgeVerdict:
            t0 = time.time()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:{reference_mime};base64,{ref_b64}"}},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{a_b64}"}},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b_b64}"}},
                            {"type": "text", "text": JUDGE_USER_TEMPLATE},
                        ],
                    },
                ],
                temperature=0.0,
                seed=self.seed,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                extra_body=extra_body or None,
            )
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Judge returned empty response")
            raw = response.choices[0].message.content.strip()
            verdict = JudgeVerdict.model_validate_json(extract_json_object(raw))
            logger.info(
                f"{prefix} Finished Task {task_id} | Elapsed: {time.time() - t0:.1f}s | "
                f"Winner: {verdict.winner} | Confidence: {verdict.confidence:.2f} | "
                f"Reason: {verdict.reason[:120]}"
            )
            return verdict

        return await async_retry(_call, max_retries=self.max_retries)
