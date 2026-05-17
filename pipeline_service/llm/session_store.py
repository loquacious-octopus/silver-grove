
from __future__ import annotations

from typing import Callable

from llm.session import SessionAgent


class SessionStore:
    """
    In-memory store of per-task, per-actor SessionAgents.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], SessionAgent] = {}

    def get_or_create(
        self,
        task_id: str,
        actor: str,
        factory: Callable[[str, str], SessionAgent],
    ) -> SessionAgent:
        key = (task_id, actor)
        sess = self._sessions.get(key)
        if sess is None:
            sess = factory(task_id, actor)
            self._sessions[key] = sess
        return sess

    def get(self, task_id: str, actor: str) -> SessionAgent | None:
        return self._sessions.get((task_id, actor))

    def evict(self, task_id: str) -> int:
        """Drop every session for this task. Returns count evicted."""
        keys = [k for k in self._sessions if k[0] == task_id]
        for k in keys:
            del self._sessions[k]
        return len(keys)

    def evict_actor(self, task_id: str, actor: str) -> bool:
        """Drop a single (task, actor) session. Returns True if found."""
        return self._sessions.pop((task_id, actor), None) is not None

    def rename_actor(
        self, task_id: str, from_actor: str, to_actor: str,
    ) -> bool:
        """Move a session from one actor key to another (same task).
        """
        sess = self._sessions.pop((task_id, from_actor), None)
        if sess is None:
            return False
        self._sessions[(task_id, to_actor)] = sess
        return True

    def __len__(self) -> int:
        return len(self._sessions)
