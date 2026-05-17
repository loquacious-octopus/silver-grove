from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from modules.js_checker.settings import JSCheckerConfig
from modules.renderer.settings import RendererConfig

_here = Path(__file__).parent.parent
config_file_dir = _here / "configuration.yaml"
if not config_file_dir.exists():
    config_file_dir = _here.parent / "configuration.yaml"



class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 10006
    debug: bool = False


class PipelineConfig(BaseModel):
    batch_time_budget: int = 7000
    prompt_timeout: float = 120.0
    use_planner: bool = True


class VllmServeConfig(BaseModel):
    """ 
    Options for launching `vllm serve` via run.sh (local endpoints only).
    """

    model: str = ""
    port: int | None = None
    gpu_ids: str = "0"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_num_seqs: int = 4
    api_key: str = "local"
    rope_patch: bool = False


class LLMClientConfig(BaseModel):
    """One named client endpoint."""
    base_url: str
    api_key_env: str = ""
    api_key: str = ""
    enabled: bool = False
    vllm: VllmServeConfig | None = None

    @property
    def backend(self) -> str:
        """Detect backend type from base_url."""
        _LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")
        url = self.base_url.lower()
        return "vllm" if any(h in url for h in _LOCAL_HOSTS) else "openrouter"

class ActorConfig(BaseModel):
    """Per-actor config."""

    workers: int = 1
    queue_size: int = 8
    client: str | None = None
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    seed: int = 42
    reasoning_effort: str | None = None
    ensemble_size: int = 1
    ensemble_temperature: float = 0.3
    multimodal: bool = False

class ActorsConfig(BaseModel):
    planner: ActorConfig = ActorConfig(
        workers=2, queue_size=8,
        client="openrouter", model="qwen/qwen2.5-vl-72b-instruct",
    )
    coder: ActorConfig = ActorConfig(
        workers=2, queue_size=8,
        client="openrouter", model="qwen/qwen-2.5-72b-instruct",
        max_tokens=8192,
        ensemble_size=4,
        ensemble_temperature=0.3,
    )
    patcher: ActorConfig = ActorConfig(workers=2, queue_size=8)
    critic: ActorConfig = ActorConfig(
        workers=3, queue_size=8,
        client="openrouter", model="qwen/qwen2.5-vl-72b-instruct",
    )
    judge: ActorConfig = ActorConfig(
        workers=3, queue_size=8,
        client="openrouter", model="qwen/qwen2.5-vl-72b-instruct",
        max_tokens=1024,
    )
    checker: ActorConfig = ActorConfig(workers=2, queue_size=8)
    renderer: ActorConfig = ActorConfig(workers=1, queue_size=4)


_LLM_ACTORS_BASE: tuple[str, ...] = ("coder", "critic", "judge")
_LLM_ACTORS_WITH_PLANNER: tuple[str, ...] = ("planner",) + _LLM_ACTORS_BASE


class EventBusConfig(BaseModel):
    max_iter: int = 2
    task_deadline_s: float = 60.0
    score_threshold: float = 0.80

def _default_llm_clients() -> dict[str, LLMClientConfig]:
    """Default LLM clients."""
    _DEFAULT_VISION = "http://localhost:8001/v1"
    _DEFAULT_CODER = "http://localhost:8002/v1"
    _DEFAULT_OPENROUTER = "https://openrouter.ai/api/v1"

    return {
        "openrouter": LLMClientConfig(
            base_url=_DEFAULT_OPENROUTER, api_key_env="OPENROUTER_API_KEY", enabled=True,
        ),
        "vision": LLMClientConfig(
            base_url=_DEFAULT_VISION, api_key_env="VLLM_API_KEY", enabled=True,
        ),
        "coder": LLMClientConfig(
            base_url=_DEFAULT_CODER, api_key_env="VLLM_API_KEY", enabled=True,
        ),
    }


class SettingsConf(BaseSettings):
    api: APIConfig = APIConfig()
    pipeline: PipelineConfig = PipelineConfig()
    benchmark: bool = True
    llm_clients: dict[str, LLMClientConfig] = Field(default_factory=_default_llm_clients)
    actors: ActorsConfig = ActorsConfig()
    event_bus: EventBusConfig = EventBusConfig()
    js_checker: JSCheckerConfig = JSCheckerConfig()
    renderer: RendererConfig = RendererConfig()

    @model_validator(mode="after")
    def _validate_actor_clients(self) -> "SettingsConf":
        known = set(self.llm_clients.keys())
        actor_names = list(
            _LLM_ACTORS_WITH_PLANNER if self.pipeline.use_planner
            else _LLM_ACTORS_BASE
        )
        if self.actors.coder.ensemble_size <= 1 and "judge" in actor_names:
            actor_names.remove("judge")
        if not self.pipeline.use_planner and not self.actors.coder.multimodal:
            raise ValueError(
                "pipeline.use_planner=false requires actors.coder.multimodal=true "
                "(coder must consume the reference image directly when there is no "
                "planner). Set actors.coder.multimodal=true and choose a "
                "vision-capable model (e.g. qwen/qwen2.5-vl-72b-instruct)."
            )
        for name in actor_names:
            actor: ActorConfig = getattr(self.actors, name)
            if actor.client is None:
                raise ValueError(
                    f"actors.{name}.client is required (must name an entry in llm_clients). "
                    f"Known clients: {sorted(known)}"
                )
            if actor.client not in known:
                raise ValueError(
                    f"actors.{name}.client={actor.client!r} does not exist in llm_clients. "
                    f"Known clients: {sorted(known)}"
                )
            if not actor.model:
                raise ValueError(
                    f"actors.{name}.model is required (non-empty string)."
                )
            client_cfg = self.llm_clients[actor.client]
            if not client_cfg.enabled:
                raise ValueError(
                    f"actors.{name}.client={actor.client!r} points to a disabled client "
                    f"(llm_clients.{actor.client}.enabled=false). Enable the client "
                    f"or change actors.{name}.client."
                )
        return self

    @model_validator(mode="after")
    def _validate_llm_client_topology(self) -> "SettingsConf":
        """Client topology: either 'openrouter' in llm_clients, or local vision+coder."""
        clients = self.llm_clients
        if "openrouter" in clients:
            return self
        if "vision" not in clients or "coder" not in clients:
            raise ValueError(
                "You must have both vision and coder clients enabled."
            )
        for label in ("vision", "coder"):
            cfg = clients[label]
            if not cfg.enabled:
                raise ValueError(
                    f"llm_clients.{label}.enabled=false is not allowed."
                )
            vllm = cfg.vllm
            if vllm is None or not (vllm.model or "").strip():
                raise ValueError(
                    f"llm_clients.{label}: set vllm.model (HF repo id) so that run.sh can start the vLLM server."
                )
        return self


def _load_yml_config(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Config not found: {path}") from error


data_yaml = _load_yml_config(config_file_dir)
settings = SettingsConf.model_validate(data_yaml)
