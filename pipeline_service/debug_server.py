from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response


LOG_DIR = Path(os.environ.get("MINI_DEBUG_LOG_DIR", "/tmp/miniature-enigma-debug"))
APP_PORT = int(os.environ.get("APP_PORT", "10006"))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "local")
APP_BASE = f"http://127.0.0.1:{APP_PORT}"
VLLM_BASE = f"http://127.0.0.1:{VLLM_PORT}"
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/workspace/hf-cache/hub"))
app = FastAPI(title="miniature-enigma debug")


def _run(command: list[str], timeout: float = 20.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-80_000:],
            "stderr": completed.stderr[-40_000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def _redact_env() -> dict[str, str]:
    out = {}
    for key, value in sorted(os.environ.items()):
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            out[key] = "<redacted>"
        elif key.startswith(("MINI_", "HF_", "HUGGING", "BENCHMARK", "CUDA", "NVIDIA", "VLLM", "APP_", "DEBUG_")):
            out[key] = value
    return out


def _log_files() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    return sorted((path for path in LOG_DIR.iterdir() if path.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def _safe_log(name: str) -> Path | None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name or ""):
        return None
    for path in _log_files():
        if path.name == name:
            return path
    return None


def _tcp_ports() -> dict[str, Any]:
    return {
        "tcp": _run(["bash", "-lc", "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null | sed -n '1,120p'"]),
        "python_probe": _run(
            [
                "python3",
                "-c",
                "import socket\n"
                "for p in [10006,10007,8001,8002,8003]:\n"
                "    s=socket.socket(); s.settimeout(1)\n"
                "    try: s.connect(('127.0.0.1',p)); print(p,'open')\n"
                "    except Exception as e: print(p,type(e).__name__,e)\n"
                "    finally: s.close()\n",
            ]
        ),
    }


async def _get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
        try:
            body: Any = response.json()
        except Exception:  # noqa: BLE001
            body = response.text[-20_000:]
        return {"status": response.status_code, "body": body}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.get("/debug/summary")
async def summary() -> JSONResponse:
    return JSONResponse(
        content={
            "app": await _get_json(f"{APP_BASE}/status", timeout=5),
            "vllm_models": await _get_json(f"{VLLM_BASE}/v1/models", headers={"Authorization": f"Bearer {VLLM_API_KEY}"}, timeout=5),
            "system": {
                "processes": _run(["ps", "-eo", "pid,ppid,stat,pcpu,pmem,etime,cmd"]),
                "nvidia_smi": _run(["nvidia-smi"]),
                "df": _run(["df", "-h"]),
                "cache_usage": _run(["du", "-sh", str(HF_CACHE.parent), str(HF_CACHE)]),
                "cache_recent": _run(["bash", "-lc", f"find {str(HF_CACHE)!r} -maxdepth 5 -type f -printf '%T@ %s %p\\n' 2>/dev/null | sort -n | tail -40"]),
                "ports": _tcp_ports(),
            },
            "logs": [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in _log_files()],
            "env": _redact_env(),
        }
    )


@app.get("/debug/system")
async def system() -> JSONResponse:
    return JSONResponse(
        content={
            "processes": _run(["ps", "-eo", "pid,ppid,stat,pcpu,pmem,etime,cmd"]),
            "nvidia_smi": _run(["nvidia-smi"]),
            "df": _run(["df", "-h"]),
            "cache_usage": _run(["du", "-sh", str(HF_CACHE.parent), str(HF_CACHE)]),
            "cache_locks": _run(["bash", "-lc", f"find {str(HF_CACHE)!r} \\( -name '*.lock' -o -name '*.incomplete' \\) -printf '%s %p\\n' 2>/dev/null | head -100"]),
            "ports": _tcp_ports(),
        }
    )


@app.get("/debug/app/status")
async def app_status() -> JSONResponse:
    return JSONResponse(content=await _get_json(f"{APP_BASE}/status", timeout=10))


@app.get("/debug/vllm/models")
async def vllm_models() -> JSONResponse:
    return JSONResponse(content=await _get_json(f"{VLLM_BASE}/v1/models", headers={"Authorization": f"Bearer {VLLM_API_KEY}"}, timeout=10))


@app.get("/debug/logs")
async def logs() -> JSONResponse:
    return JSONResponse(content={"logs": [{"name": p.name, "path": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in _log_files()]})


@app.get("/debug/logs/{name}")
async def log_tail(name: str, max_bytes: int = 200_000) -> JSONResponse:
    path = _safe_log(name)
    if path is None:
        raise HTTPException(status_code=404, detail="log file not found")
    raw = path.read_bytes()
    data = raw[-max_bytes:]
    return JSONResponse(content={"name": path.name, "size": len(raw), "truncated": len(raw) > max_bytes, "text": data.decode(errors="replace")})
