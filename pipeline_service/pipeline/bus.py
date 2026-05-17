
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from logger_config import logger
from pipeline.events import Event, TaskDone, TaskFailed


Handler = Callable[[Event], Awaitable[None]]


@dataclass
class _ActorBinding:
    name: str
    queue: asyncio.Queue
    handler: Handler
    workers: list[asyncio.Task] = field(default_factory=list)
    worker_count: int = 1


class EventBus:
    """Pub/sub bus with per-actor queues and per-event-type routing."""

    def __init__(self) -> None:
        self._actors: dict[str, _ActorBinding] = {}
        # event_type → [actor_name, ...]
        self._routes: dict[str, list[str]] = defaultdict(list)
        self._running = False

    # Actor registration

    def register_actor(
        self,
        name: str,
        handler: Handler,
        *,
        workers: int = 1,
        queue_size: int = 8,
    ) -> None:
        """Register a handler coroutine and its queue."""
        if name in self._actors:
            raise ValueError(f"actor already registered: {name}")
        self._actors[name] = _ActorBinding(
            name=name,
            queue=asyncio.Queue(maxsize=queue_size),
            handler=handler,
            worker_count=max(1, workers),
        )

    def subscribe(self, event_type: str, actor: str) -> None:
        """Route events of `event_type` into `actor`'s queue."""
        if actor not in self._actors:
            raise ValueError(f"unknown actor: {actor}")
        if actor not in self._routes[event_type]:
            self._routes[event_type].append(actor)

    # Lifecycle

    async def start(self) -> None:
        if self._running:
            return
        for binding in self._actors.values():
            for i in range(binding.worker_count):
                task = asyncio.create_task(
                    self._drain(binding, i),
                    name=f"bus.{binding.name}.w{i}",
                )
                binding.workers.append(task)
        self._running = True
        logger.info(f"event_bus started | actors={list(self._actors)}")

    async def stop(self) -> None:
        if not self._running:
            return
        for binding in self._actors.values():
            for task in binding.workers:
                task.cancel()
            for task in binding.workers:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            binding.workers.clear()
        self._running = False
        logger.info("event_bus stopped")

    # Dispatch

    async def publish(self, event: Event) -> None:
        """Route an event to every subscribed actor's queue."""
        event_type = event.type
        actors = self._routes.get(event_type, [])
        if not actors:
            logger.debug(
                f"event without subscribers: type={event_type} task={event.task_id}"
            )
            return
        for actor_name in actors:
            binding = self._actors.get(actor_name)
            if binding is None:
                continue
            await binding.queue.put(event)

    async def _drain(self, binding: _ActorBinding, worker_index: int) -> None:
        log_prefix = f"bus.{binding.name}.w{worker_index}"
        while True:
            event = await binding.queue.get()
            try:
                await binding.handler(event)
            except Exception as exc:
                logger.exception(
                    f"{log_prefix} handler crash task={event.task_id} "
                    f"type={event.type} err={exc}"
                )
                await self.publish(TaskFailed(
                    task_id=event.task_id,
                    error=f"handler {binding.name} crashed: {type(exc).__name__}: {exc}",
                    stage=binding.name,
                ))
            finally:
                binding.queue.task_done()

    # Introspection (for tests / debug)

    @property
    def routes(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._routes.items()}

    @property
    def actors(self) -> list[str]:
        return list(self._actors)

    def queue(self, actor: str) -> asyncio.Queue:
        return self._actors[actor].queue
