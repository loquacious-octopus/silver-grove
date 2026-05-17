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


LOG_DIR = Path(os.environ.get("PROBE_LOG_DIR", "/tmp/qwen35-probe-logs"))
MANIFEST_FILE = Path(os.environ.get("STARTUP_MANIFEST_FILE", LOG_DIR / "startup_manifest.json"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-397B-A17B-FP8")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_ID)
REQUIRED_MODEL_SUBSTRING = os.environ.get("REQUIRED_MODEL_SUBSTRING", SERVED_MODEL_NAME)
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "local")
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"
APP_PORT = int(os.environ.get("APP_PORT", "10006"))
APP_BASE_URL = f"http://127.0.0.1:{APP_PORT}"

app = FastAPI(title="Qwen3.5 startup probe")


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_FILE.exists():
        return {"exists": False, "path": str(MANIFEST_FILE)}
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "path": str(MANIFEST_FILE), "error": repr(exc)}


def _log_files() -> list[Path]:
    files = []
    if LOG_DIR.exists():
        files.extend(path for path in LOG_DIR.iterdir() if path.is_file())
    if MANIFEST_FILE.exists() and MANIFEST_FILE not in files:
        files.append(MANIFEST_FILE)
    return sorted(files, key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)


def _safe_log_path(name: str) -> Path | None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name or ""):
        return None
    for path in _log_files():
        if path.name == name:
            return path
    return None


async def _get_vllm_json(path: str, timeout: float = 10.0) -> tuple[int, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{VLLM_BASE_URL}{path}", headers={"Authorization": f"Bearer {VLLM_API_KEY}"})
        try:
            body: Any = response.json()
        except Exception:  # noqa: BLE001
            body = response.text[-20_000:]
        return response.status_code, body
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": repr(exc)}


async def _post_vllm_json(path: str, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}{path}",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                json=payload,
            )
        try:
            body: Any = response.json()
        except Exception:  # noqa: BLE001
            body = response.text[-20_000:]
        return response.status_code, body
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": repr(exc)}


async def _get_app_json(path: str, timeout: float = 30.0) -> tuple[int, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{APP_BASE_URL}{path}")
        try:
            body: Any = response.json()
        except Exception:  # noqa: BLE001
            body = response.text[-20_000:]
        return response.status_code, body
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": repr(exc)}


async def _post_app_json(path: str, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{APP_BASE_URL}{path}", json=payload)
        try:
            body: Any = response.json()
        except Exception:  # noqa: BLE001
            body = response.text[-20_000:]
        return response.status_code, body
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": repr(exc)}


def _run_text(command: list[str], timeout: float = 15.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-40_000:],
            "stderr": completed.stderr[-20_000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.get("/ready")
async def ready() -> JSONResponse:
    status, body = await _get_vllm_json("/v1/models")
    body_text = json.dumps(body, sort_keys=True)
    is_ready = status == 200 and REQUIRED_MODEL_SUBSTRING in body_text
    payload = {
        "ready": is_ready,
        "vllm_status": status,
        "required_model_substring": REQUIRED_MODEL_SUBSTRING,
        "models": body,
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=payload)


@app.get("/debug/manifest")
async def debug_manifest() -> JSONResponse:
    return JSONResponse(content=_load_manifest())


@app.get("/debug/system")
async def debug_system() -> JSONResponse:
    cache_root = Path(os.environ.get("HF_HUB_CACHE", "/workspace/hf-cache/hub"))
    return JSONResponse(
        content={
            "processes": _run_text(
                [
                    "ps",
                    "-eo",
                    "pid,ppid,stat,pcpu,pmem,etime,cmd",
                ]
            ),
            "nvidia_smi": _run_text(["nvidia-smi"]),
            "cache_usage": _run_text(["du", "-sh", str(cache_root.parent), str(cache_root)]),
            "cache_files": _run_text(
                [
                    "bash",
                    "-lc",
                    f"find {str(cache_root)!r} -maxdepth 4 -type f 2>/dev/null | wc -l",
                ]
            ),
        }
    )


@app.get("/debug/logs")
async def debug_logs() -> JSONResponse:
    logs = []
    for path in _log_files():
        stat = path.stat()
        logs.append(
            {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "updated_epoch": stat.st_mtime,
            }
        )
    return JSONResponse(content={"logs": logs})


@app.get("/debug/logs/{name}")
async def debug_log(name: str, max_bytes: int = 200_000) -> JSONResponse:
    if max_bytes <= 0:
        raise HTTPException(status_code=400, detail="max_bytes must be positive")
    path = _safe_log_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="log file not found")
    raw = path.read_bytes()
    payload = raw[-max_bytes:]
    return JSONResponse(
        content={
            "name": path.name,
            "path": str(path),
            "size": len(raw),
            "max_bytes": max_bytes,
            "truncated": len(raw) > max_bytes,
            "text": payload.decode(errors="replace"),
        }
    )


@app.post("/debug/chat-test")
async def debug_chat_test() -> JSONResponse:
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with exactly READY."}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, body = await _post_vllm_json("/v1/chat/completions", payload, timeout=120.0)
    body_text = json.dumps(body, sort_keys=True)
    ok = status == 200 and "choices" in body_text
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "ok": ok,
            "vllm_status": status,
            "request": payload,
            "response": body,
        },
    )


@app.get("/debug/app/status")
async def debug_app_status() -> JSONResponse:
    status, body = await _get_app_json("/status")
    return JSONResponse(status_code=200 if status else 503, content={"status": status, "body": body})


@app.get("/debug/app/tasks")
async def debug_app_tasks() -> JSONResponse:
    status, body = await _get_app_json("/debug/tasks")
    return JSONResponse(status_code=200 if status else 503, content={"status": status, "body": body})


@app.get("/debug/app/tasks/{stem}")
async def debug_app_task(stem: str) -> JSONResponse:
    status, body = await _get_app_json(f"/debug/tasks/{stem}")
    return JSONResponse(status_code=200 if status else 503, content={"status": status, "body": body})


@app.post("/debug/app/generate")
async def debug_app_generate(payload: dict[str, Any]) -> JSONResponse:
    status, body = await _post_app_json("/generate", payload, timeout=120.0)
    return JSONResponse(status_code=200 if status else 503, content={"status": status, "body": body})
