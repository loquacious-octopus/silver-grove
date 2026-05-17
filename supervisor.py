from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path(os.environ.get("PROBE_LOG_DIR", "/tmp/qwen35-probe-logs"))
MANIFEST_FILE = Path(os.environ.get("STARTUP_MANIFEST_FILE", LOG_DIR / "startup_manifest.json"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-397B-A17B-FP8")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_ID)
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
DEBUG_PORT = int(os.environ.get("DEBUG_PORT", "10007"))
APP_PORT = int(os.environ.get("APP_PORT", "10006"))
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "local")
VLLM_BIN = os.environ.get("VLLM_BIN", "/opt/vllm-env/bin/vllm")
SUPERVISOR_LOG = LOG_DIR / "supervisor.log"
VLLM_LOG = LOG_DIR / "vllm.log"
DEBUG_LOG = LOG_DIR / "debug_server.log"
APP_LOG = LOG_DIR / "batch_service.log"

processes: list[subprocess.Popen] = []


def _log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run_text(command: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=timeout)[-20_000:]
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


def _vllm_command() -> list[str]:
    command = [
        VLLM_BIN,
        "serve",
        MODEL_ID,
        "--host",
        os.environ.get("VLLM_HOST", "0.0.0.0"),
        "--port",
        str(VLLM_PORT),
        "--api-key",
        VLLM_API_KEY,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        os.environ.get("TENSOR_PARALLEL_SIZE", "4"),
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-model-len",
        os.environ.get("MAX_MODEL_LEN", "65536"),
        "--max-num-seqs",
        os.environ.get("MAX_NUM_SEQS", "1"),
        "--max-num-batched-tokens",
        os.environ.get("MAX_NUM_BATCHED_TOKENS", "32768"),
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--dtype",
        "auto",
    ]
    extra_args = os.environ.get("VLLM_EXTRA_ARGS", "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


def _write_manifest(command: list[str]) -> None:
    env_keys = [
        "MODEL_ID",
        "SERVED_MODEL_NAME",
        "TENSOR_PARALLEL_SIZE",
        "GPU_MEMORY_UTILIZATION",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
        "VLLM_EXTRA_ARGS",
        "APP_PORT",
        "DEBUG_PORT",
        "VLLM_PORT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "CUDA_VISIBLE_DEVICES",
    ]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_command": command,
        "environment": {key: os.environ.get(key) for key in env_keys},
        "python": _run_text([sys.executable, "--version"]),
        "vllm_version": _run_text([VLLM_BIN, "--version"]),
        "nvidia_smi": _run_text(["nvidia-smi"]),
        "disk": _run_text(["df", "-h"]),
        "logs": {
            "supervisor": str(SUPERVISOR_LOG),
            "vllm": str(VLLM_LOG),
            "debug": str(DEBUG_LOG),
            "app": str(APP_LOG),
        },
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _start(command: list[str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("ab")
    _log(f"starting: {' '.join(shlex.quote(part) for part in command)}")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    processes.append(process)
    return process


def _shutdown(signum: int, _frame: object) -> None:
    _log(f"received signal {signum}; terminating children")
    for process in processes:
        if process.poll() is None:
            process.terminate()
    time.sleep(5)
    for process in processes:
        if process.poll() is None:
            process.kill()
    raise SystemExit(128 + signum)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    command = _vllm_command()
    _write_manifest(command)
    _log(f"startup manifest written: {MANIFEST_FILE}")

    debug = _start(
        [sys.executable, "-m", "uvicorn", "debug_server:app", "--host", "0.0.0.0", "--port", str(DEBUG_PORT)],
        DEBUG_LOG,
    )
    app = _start(
        [sys.executable, "-m", "uvicorn", "batch_service:app", "--host", "0.0.0.0", "--port", str(APP_PORT)],
        APP_LOG,
    )
    vllm = _start(command, VLLM_LOG)

    vllm_exit_logged = False
    while True:
        if debug.poll() is not None:
            _log(f"debug server exited with code {debug.returncode}; stopping container")
            return debug.returncode or 1
        if app.poll() is not None:
            _log(f"batch app exited with code {app.returncode}; stopping container")
            return app.returncode or 1
        if vllm.poll() is not None and not vllm_exit_logged:
            _log(f"vLLM exited with code {vllm.returncode}; keeping app/debug servers alive for log collection")
            vllm_exit_logged = True
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
