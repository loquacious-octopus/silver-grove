from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx

from logger_config import logger
from modules.base import BaseModule
from modules.renderer.settings import RendererConfig
from pipeline.task import PipelineTask

_RUNNER_JS = Path(__file__).parent / "render_service" / "render_runner.mjs"


class RendererModule(BaseModule):

    def __init__(self, config: RendererConfig) -> None:
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None

    async def startup(self) -> None:
        if not _RUNNER_JS.exists():
            logger.warning(f"[RENDERER] runner missing at {_RUNNER_JS}, disabling")
            return

        node_cwd = self._resolve_node_cwd()
        logger.info(
            f"[RENDERER] starting sidecar | node={self.config.node_binary} "
            f"runner={_RUNNER_JS} port={self.config.sidecar_port} cwd={node_cwd}"
        )

        env = {**os.environ, "PORT": str(self.config.sidecar_port)}
        self._proc = await asyncio.create_subprocess_exec(
            self.config.node_binary,
            str(_RUNNER_JS),
            env=env,
            cwd=node_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._stdout_task = asyncio.create_task(
            self._pipe_logger(self._proc.stdout, "stdout")
        )
        self._stderr_task = asyncio.create_task(
            self._pipe_logger(self._proc.stderr, "stderr")
        )

        self._client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
        await self._wait_ready()
        logger.info(f"[RENDERER] sidecar ready on {self._base_url}")

    async def shutdown(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

        if self._proc is not None and self._proc.returncode is None:
            logger.info("[RENDERER] terminating sidecar")
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("[RENDERER] sidecar did not exit on SIGTERM, sending SIGKILL")
                    self._proc.kill()
                    await self._proc.wait()
            except ProcessLookupError:
                pass

        for t in (self._stdout_task, self._stderr_task):
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

        self._proc = None
        self._stdout_task = None
        self._stderr_task = None

    async def process(self, task: PipelineTask) -> PipelineTask:
        if task.failed or not task.js_code:
            logger.debug(f"[RENDERER] '{task.stem}' skip (failed={task.failed}, has_code={bool(task.js_code)})")
            return task

        if self._client is None:
            task.render_errors = ["sidecar not started"]
            logger.warning(f"[RENDERER] '{task.stem}' skip — sidecar not started")
            return task

        logger.info(f"[RENDERER] '{task.stem}' start | js_code={len(task.js_code)} bytes")

        payload = {
            "source": task.js_code,
            "options": {
                "imgSize": self.config.img_size,
                "gap": self.config.grid_gap,
            },
        }
        if self.config.bg_color:
            payload["options"]["bgColor"] = self.config.bg_color

        t0 = time.monotonic()
        try:
            resp = await self._client.post(f"{self._base_url}/render/grid", json=payload)
        except Exception as exc:
            task.render_errors = [f"{type(exc).__name__}: {exc}"]
            logger.warning(f"[RENDERER] '{task.stem}' FAIL (http) | {task.render_errors[0]}")
            return task

        task.render_ms = (time.monotonic() - t0) * 1000.0

        if resp.status_code == 200:
            task.rendered_png = resp.content
            logger.info(
                f"[RENDERER] '{task.stem}' PASS grid | "
                f"png={len(task.rendered_png)}B render={task.render_ms/1000:.1f}s"
            )
        else:
            detail = resp.text[:200] if resp.text else ""
            task.render_errors = [f"HTTP {resp.status_code}: {detail}"]
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (status) | "
                f"{task.render_errors[0]} | render={task.render_ms/1000:.1f}s"
            )
            return task

        return task

    @property
    def _base_url(self) -> str:
        return f"http://{self.config.sidecar_host}:{self.config.sidecar_port}"

    def _resolve_node_cwd(self) -> str:
        preferred = os.environ.get("NODE_CWD")
        candidates = [preferred, "/workspace", str(_RUNNER_JS.parent.parent.parent.parent)]
        for c in candidates:
            if c and os.path.isdir(os.path.join(c, "node_modules")):
                return c
        return str(_RUNNER_JS.parent)

    async def _wait_ready(self) -> None:
        assert self._client is not None
        deadline = time.monotonic() + self.config.startup_timeout_s
        last_err: str = ""
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.returncode is not None:
                raise RuntimeError(f"[RENDERER] sidecar exited early: rc={self._proc.returncode}")
            try:
                r = await self._client.get(f"{self._base_url}/ping", timeout=2.0)
                if r.status_code == 200:
                    return
                last_err = f"status={r.status_code}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)
        raise RuntimeError(
            f"[RENDERER] sidecar not ready after {self.config.startup_timeout_s}s "
            f"(last: {last_err})"
        )

    @staticmethod
    async def _pipe_logger(stream: asyncio.StreamReader | None, which: str) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    if which == "stderr":
                        logger.warning(f"[renderer-sidecar] {text}")
                    else:
                        logger.debug(f"[renderer-sidecar] {text}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"[renderer-sidecar] pipe {which} error: {exc}")
