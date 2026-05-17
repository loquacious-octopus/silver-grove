
from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from modules.critic.schema import CriticReport, Issue


class _EventBase(BaseModel):
    task_id: str
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)


class TaskCreated(_EventBase):
    type: Literal["task.created"] = "task.created"
    image_url: str
    seed: int = 42


class PlannerDone(_EventBase):
    type: Literal["planner.done"] = "planner.done"
    osd: dict[str, Any]


class CoderDone(_EventBase):
    type: Literal["coder.done"] = "coder.done"
    js_code: str
    attempt: int = 1


class CheckerOk(_EventBase):
    type: Literal["checker.ok"] = "checker.ok"
    js_code: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class CheckerFailed(_EventBase):
    type: Literal["checker.failed"] = "checker.failed"
    js_errors: list[str] = Field(default_factory=list)


class RenderDone(_EventBase):
    type: Literal["render.done"] = "render.done"
    rendered_png: bytes | None = None


class CriticDone(_EventBase):
    type: Literal["critic.done"] = "critic.done"
    report: CriticReport


class PatcherDone(_EventBase):
    type: Literal["patcher.done"] = "patcher.done"
    js_code: str
    applied_ops: list[str] = Field(default_factory=list)
    iteration: int = 1


class TaskDone(_EventBase):
    type: Literal["task.done"] = "task.done"
    artifact: dict[str, Any] = Field(default_factory=dict)   # {"js": str, "views": {...}}


class TaskFailed(_EventBase):
    type: Literal["task.failed"] = "task.failed"
    error: str
    stage: str | None = None


Event = Union[
    TaskCreated,
    PlannerDone,
    CoderDone,
    CheckerOk,
    CheckerFailed,
    RenderDone,
    CriticDone,
    PatcherDone,
    TaskDone,
    TaskFailed,
]
