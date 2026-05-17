#!/bin/bash
set -e

CONFIG_FILE="${CONFIG_PATH:-/workspace/configuration.yaml}"
export CONFIG_FILE
set -o pipefail
MINI_DEBUG_LOG_DIR="${MINI_DEBUG_LOG_DIR:-/tmp/miniature-enigma-debug}"
DEBUG_PORT="${DEBUG_PORT:-10007}"
APP_PORT="${APP_PORT:-10006}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_API_KEY="${VLLM_API_KEY:-local}"
export MINI_DEBUG_LOG_DIR DEBUG_PORT APP_PORT VLLM_PORT VLLM_API_KEY
mkdir -p "$MINI_DEBUG_LOG_DIR"
echo "=== STARTING DEBUG SERVER on ${DEBUG_PORT} ==="
python -m uvicorn debug_server:app --host 0.0.0.0 --port "$DEBUG_PORT" > >(tee -a "$MINI_DEBUG_LOG_DIR/debug_server.log") 2>&1 &
DEBUG_PID=$!


python3 <<'PY'
import os
from pathlib import Path
import yaml

path = Path(os.environ.get("CONFIG_FILE", "/workspace/configuration.yaml"))
cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

client_name = os.environ.get("MINI_VLLM_CLIENT", "vision-coder")
client = cfg.setdefault("llm_clients", {}).setdefault(client_name, {})
vllm = client.setdefault("vllm", {})

string_fields = {
    "MINI_GPU_IDS": "gpu_ids",
}
int_fields = {
    "MINI_TENSOR_PARALLEL_SIZE": "tensor_parallel_size",
    "MINI_MAX_MODEL_LEN": "max_model_len",
    "MINI_MAX_NUM_SEQS": "max_num_seqs",
    "MINI_MAX_NUM_BATCHED_TOKENS": "max_num_batched_tokens",
}
float_fields = {
    "MINI_GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
}

changed = False
for env_name, key in string_fields.items():
    value = os.environ.get(env_name)
    if value:
        vllm[key] = value
        changed = True
for env_name, key in int_fields.items():
    value = os.environ.get(env_name)
    if value:
        vllm[key] = int(value)
        changed = True
for env_name, key in float_fields.items():
    value = os.environ.get(env_name)
    if value:
        vllm[key] = float(value)
        changed = True

if changed:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"=== HARDWARE OVERRIDES APPLIED: {client_name} {vllm} ===", flush=True)
else:
    print("=== HARDWARE OVERRIDES: none ===", flush=True)
PY



if python -m modules.metrics.preflight; then
    echo "=== PRE-FLIGHT OK ==="
else
    echo "=== PRE-FLIGHT FAILED — starting FastAPI in REPLACE mode, skipping vLLM ==="
    python serve.py 2>&1 | tee -a "$MINI_DEBUG_LOG_DIR/app.log"
    exit ${PIPESTATUS[0]}
fi


echo "STARTING MAIN FASTAPI SERVICE"
python serve.py > >(tee -a "$MINI_DEBUG_LOG_DIR/app.log") 2>&1 &
SERVE_PID=$!


eval "$(python3 <<PY
import os, re, shlex, sys
import yaml

path = os.environ.get("CONFIG_FILE", "/workspace/configuration.yaml")
try:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"echo 'WARNING: Could not read config: {e}' >&2", file=sys.stderr)
    cfg = {}

llm = cfg.get("llm_clients") or {}
jobs = []

def _is_local(url):
    u = (url or "").lower()
    return any(h in u for h in ("localhost", "127.0.0.1", "0.0.0.0"))

def _port(base_url, override):
    if override is not None:
        return int(override)
    m = re.search(r":(\d+)(?:/|$)", base_url or "")
    return int(m.group(1)) if m else 8001

for name, spec in llm.items():
    if not isinstance(spec, dict):
        continue
    if spec.get("enabled", True) is False:
        continue
    v = spec.get("vllm") or {}
    if not isinstance(v, dict):
        continue
    model = (v.get("model") or "").strip()
    if not model:
        continue
    base = spec.get("base_url") or ""
    if not _is_local(base):
        continue
    jobs.append({
        "name": name, "model": model,
        "port": _port(base, v.get("port")),
        "gpu_ids": str(v.get("gpu_ids", "0")),
        "tp": int(v.get("tensor_parallel_size", 1)),
        "gpu_util": float(v.get("gpu_memory_utilization", 0.90)),
        "max_len": int(v.get("max_model_len", 8192)),
        "max_seqs": int(v.get("max_num_seqs", 4)),
        "api_key": str(v.get("api_key", "local")),
        "rope": 1 if v.get("rope_patch") else 0,
    })

print(f"VLLM_START_COUNT={len(jobs)}")
for i, j in enumerate(jobs):
    print(f"VLLM_{i}_NAME={shlex.quote(j['name'])}")
    print(f"VLLM_{i}_MODEL={shlex.quote(j['model'])}")
    print(f"VLLM_{i}_PORT={j['port']}")
    print(f"VLLM_{i}_GPU_IDS={shlex.quote(j['gpu_ids'])}")
    print(f"VLLM_{i}_TP={j['tp']}")
    print(f"VLLM_{i}_GPU_UTIL={j['gpu_util']}")
    print(f"VLLM_{i}_MAX_LEN={j['max_len']}")
    print(f"VLLM_{i}_MAX_SEQS={j['max_seqs']}")
    print(f"VLLM_{i}_API_KEY={shlex.quote(j['api_key'])}")
    print(f"VLLM_{i}_ROPE={j['rope']}")
PY
)"

rope_patch_model() {
    local model="$1"
    export ROPE_PATCH_MODEL="$model"
    echo "[rope-patch] Ensuring model is downloaded and config patched: $model"
    /opt/vllm-env/bin/python3 <<'PYEOF'
import os, json, sys
from huggingface_hub import snapshot_download

model = os.environ["ROPE_PATCH_MODEL"]
print(f"[rope-patch] Downloading {model} (if needed)...", flush=True)
path = snapshot_download(repo_id=model, allow_patterns=["config.json", "*.json", "*.safetensors", "*.txt", "*.py", "tokenizer*"])
print(f"[rope-patch] Model path: {path}", flush=True)

config_file = os.path.join(path, "config.json")
if not os.path.exists(config_file):
    print(f"[rope-patch] WARNING: config.json not found at {config_file}", flush=True)
    sys.exit(0)

cfg = json.loads(open(config_file).read())
rs = cfg.get("rope_scaling", {})
print(f"[rope-patch] Original rope_scaling: {rs}", flush=True)

if "type" in rs and "rope_type" in rs:
    rs.pop("type")
    rs["rope_type"] = "mrope"
    open(config_file, "w").write(json.dumps(cfg, indent=2))
    print(f"[rope-patch] Patched {config_file}", flush=True)
elif "type" in rs and rs.get("type") == "mrope":
    rs.pop("type")
    rs["rope_type"] = "mrope"
    open(config_file, "w").write(json.dumps(cfg, indent=2))
    print(f"[rope-patch] Patched (legacy) {config_file}", flush=True)
else:
    print(f"[rope-patch] No patch needed", flush=True)
PYEOF
}

start_one_vllm() {
    local idx="$1"
    eval "local n=\$VLLM_${idx}_NAME"
    eval "local model=\$VLLM_${idx}_MODEL"
    eval "local port=\$VLLM_${idx}_PORT"
    eval "local gpus=\$VLLM_${idx}_GPU_IDS"
    eval "local tp=\$VLLM_${idx}_TP"
    eval "local util=\$VLLM_${idx}_GPU_UTIL"
    eval "local maxlen=\$VLLM_${idx}_MAX_LEN"
    eval "local maxseq=\$VLLM_${idx}_MAX_SEQS"
    eval "local apikey=\$VLLM_${idx}_API_KEY"
    eval "local rope=\$VLLM_${idx}_ROPE"

    echo "Starting vLLM | client=$n | model=$model | port=$port | GPUs=$gpus | TP=$tp"
    if [ "$rope" = "1" ]; then
        rope_patch_model "$model"
    fi

    CUDA_VISIBLE_DEVICES=$gpus /opt/vllm-env/bin/vllm serve "$model" \
        --port "$port" \
        --api-key "$apikey" \
        --max-model-len "$maxlen" \
        --tensor-parallel-size "$tp" \
        --gpu-memory-utilization "$util" \
        --max_num_seqs "$maxseq" \
        --reasoning-parser qwen3 \
        --generation-config vllm \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-num-batched-tokens 8192 2>&1 | tee -a "$MINI_DEBUG_LOG_DIR/vllm_${n}.log"
}

if [ "${VLLM_START_COUNT:-0}" -gt 0 ]; then
    echo "Starting ${VLLM_START_COUNT} vLLM instance(s) from llm_clients config..."
    i=0
    while [ "$i" -lt "$VLLM_START_COUNT" ]; do
        start_one_vllm "$i"
        i=$((i + 1))
    done
    echo "vLLM instance(s) launched in background — readiness tracked by FastAPI's OrchestratorChecker."
else
    echo "No local vLLM instances configured — skipping."
fi

wait $SERVE_PID
