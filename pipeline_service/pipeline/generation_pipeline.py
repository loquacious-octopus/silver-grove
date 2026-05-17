from __future__ import annotations

import asyncio
import os

import httpx
from openai import AsyncOpenAI

from config.settings import LLMClientConfig, SettingsConf
from logger_config import logger
from modules.js_checker.module import JSCheckerModule
from modules.renderer.module import RendererModule
from pipeline.pipeline_factory import Pipeline, build_pipeline
from pipeline.state import MinerState, MinerStatus
from pipeline.task import PipelineTask

_PLACEHOLDER_KEYS = {"", "placeholder", "your-key-here", "sk-or-...", "changeme"}
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


def _is_local_endpoint(base_url: str) -> bool:
    return any(h in base_url for h in _LOCAL_HOSTS)


def _resolve_api_key(name: str, cfg: LLMClientConfig) -> str | None:
    """Resolve api key for one client"""
    raw = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        api_key = cfg.api_key.strip()

    if _is_local_endpoint(cfg.base_url):
        return api_key or "local"

    if api_key.lower() in _PLACEHOLDER_KEYS:
        env_hint = f" (env var {cfg.api_key_env})" if cfg.api_key_env else ""
        logger.warning(
            f"Skipping LLM client {name!r}{env_hint}: placeholder/empty API key "
        )
        return None
    return api_key


class GenerationPipeline:
    """Top-level pipeline driver. Constructed once per app lifecycle."""

    def __init__(self, settings: SettingsConf, state: MinerState) -> None:
        self.settings = settings
        self.state = state

        # DET modules
        self.js_checker = JSCheckerModule(settings.js_checker)
        self.renderer = RendererModule(settings.renderer)

        self._clients: dict[str, AsyncOpenAI] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._pipeline: Pipeline | None = None

    # Lifecycle

    async def startup(self) -> None:
        logger.info("GenerationPipeline starting")

        for name, cfg in self.settings.llm_clients.items():
            if not cfg.enabled:
                continue
            api_key = _resolve_api_key(name, cfg)
            if api_key is None:
                continue
            self._clients[name] = AsyncOpenAI(
                base_url=cfg.base_url,
                api_key=api_key,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
            logger.info(
                f"llm client ready | name={name} base_url={cfg.base_url} "
                f"local={_is_local_endpoint(cfg.base_url)}"
            )

        self._http_client = httpx.AsyncClient(timeout=30.0)

        await self.js_checker.startup()
        await self.renderer.startup()

        self._pipeline = build_pipeline(
            settings=self.settings,
            clients=self._clients,
            js_checker=self.js_checker,
            renderer=self.renderer,
            http_client=self._http_client,
        )
        await self._pipeline.start()
        actors = self.settings.actors
        eb = self.settings.event_bus
        logger.info(f"[Pipeline initialized — waiting for models]")
        logger.info(f"Planner: {actors.planner.client} | Model: {actors.planner.model}")
        logger.info(f"Coder: {actors.coder.client} | Model: {actors.coder.model}")
        logger.info(f"Critic: {actors.critic.client} | Model: {actors.critic.model}")
        logger.info(f"Iter cap: {eb.max_iter} | Deadline: {eb.task_deadline_s:.0f}s | Threshold: {eb.score_threshold:.2f}")

    async def shutdown(self) -> None:
        logger.info("[Pipeline shutting down]")
        if self._pipeline is not None:
            await self._pipeline.stop()
        await self.renderer.shutdown()
        await self.js_checker.shutdown()
        if self._http_client is not None:
            await self._http_client.aclose()
        for name, client in list(self._clients.items()):
            try:
                await client.close()
            except Exception as exc:
                logger.warning(f"[Pipeline shutdown] Client {name} close failed: {exc}")

    # Batch

    async def run_batch(self, tasks: list[PipelineTask]) -> None:
        if self._pipeline is None:
            logger.error("[Batch failed] Run called before startup")
            return
        try:
            logger.info(f"[Batch starting] {len(tasks)} tasks")
            futures = [await self._pipeline.submit(t) for t in tasks]
            for finish in asyncio.as_completed(futures):
                task = await finish
                self.state.record_task(task)
                logger.info(f"[Batch progress] {self.state.progress}/{len(tasks)} tasks completed")
        except asyncio.CancelledError:
            logger.info("[Batch cancelled]")
            raise
        except Exception as exc:
            logger.exception(f"[Batch failed] {exc}")
        finally:
            self.state.status = MinerStatus.COMPLETE
            logger.info(
                f"[Batch done] {len(self.state.results)} ok, "
                f"{len(self.state.failed)} failed"
            )
    
    async def run_warmup(self, task: PipelineTask) -> None:
        if self._pipeline is None:
            logger.error("[Warmup failed] Run called before startup")
            return
        try:
            logger.info(f"[Warmup task starting]")
            future = await self._pipeline.submit(task)
            await future
            logger.info(f"[Warmup task completed]")
        except asyncio.CancelledError:
            logger.info("[Warmup task cancelled]")
            raise
        except Exception as exc:
            logger.exception(f"[Warmup failed] {exc}")
