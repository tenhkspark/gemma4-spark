#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
env_file="${GEMMA4_ENV:-gemma4-v2.env}"
rest=()
while [ $# -gt 0 ]; do
  case "$1" in
    --env)
      [ $# -ge 2 ] || { echo "FAIL: --env needs a file argument" >&2; exit 1; }
      env_file="$2"; shift 2 ;;
    *) rest+=("$1"); shift ;;
  esac
done
set -- ${rest[@]+"${rest[@]}"}
case "$env_file" in /*) ;; *) env_file="$DIR/$env_file" ;; esac
[ -f "$env_file" ] || { echo "FAIL: env file not found: $env_file" >&2; exit 1; }
. "$env_file"

usage() {
  cat <<EOF
usage: gemma4-v2-serve.sh <up|down|status|smoke> [--env <file>]
  up      docker run $GEMMA4_CONTAINER ($GEMMA4_IMAGE, $GEMMA4_MODEL,
          port $GEMMA4_PORT, MTP=$GEMMA4_MTP, KV=$GEMMA4_KV), then wait READY
  down    remove the container (other containers untouched)
  status  container state + API check + last log line
  smoke   one Japanese question, prints tok/s
EOF
  exit 1
}

container_state() { docker inspect -f '{{.State.Status}}' "$GEMMA4_CONTAINER" 2>/dev/null || echo missing; }
api_ok() { curl -s --max-time 5 "http://127.0.0.1:$GEMMA4_PORT/v1/models" | grep -q '"id"'; }

cmd_up() {
  case "$GEMMA4_MTP" in 0|2|3|4|6|8) ;; *) echo "FAIL: GEMMA4_MTP must be 0|2|3|4|6|8" >&2; exit 1;; esac
  case "$GEMMA4_KV" in auto|fp8) ;; *) echo "FAIL: GEMMA4_KV must be auto|fp8" >&2; exit 1;; esac

  local kv_arg="" spec_arg="" model_mnt model_ref mtp_mnt="" jit_mnt="" jit_env=""
  local prefix_cache_arg="--enable-prefix-caching"
  [ "${GEMMA4_PREFIX_CACHE:-1}" = 0 ] && prefix_cache_arg="--no-enable-prefix-caching"
  local lmo_arg=""
  [ "${GEMMA4_LMO:-0}" = 1 ] && lmo_arg="--language-model-only"
  [ "$GEMMA4_KV" = fp8 ] && kv_arg="--kv-cache-dtype fp8"
  if [ "$GEMMA4_MTP" != 0 ]; then
    [ -n "${GEMMA4_MTP_DIR:-}" ] \
      || { echo "FAIL: GEMMA4_MTP_DIR is unset but GEMMA4_MTP=$GEMMA4_MTP" >&2; exit 1; }
    local mtp_ref
    case "$GEMMA4_MTP_DIR" in
      /*) [ -d "$GEMMA4_MTP_DIR" ] || { echo "FAIL: MTP dir not found: $GEMMA4_MTP_DIR" >&2; exit 1; }
          mtp_mnt="-v $GEMMA4_MTP_DIR:/checkpoint-mtp:ro"; mtp_ref=/checkpoint-mtp ;;
      *)  mtp_ref="$GEMMA4_MTP_DIR" ;;
    esac
    spec_arg="--speculative-config '{\"method\":\"mtp\",\"model\":\"$mtp_ref\",\"num_speculative_tokens\":$GEMMA4_MTP}'"
  fi
  case "$GEMMA4_MODEL" in
    /*) [ -d "$GEMMA4_MODEL" ] || { echo "FAIL: model dir not found: $GEMMA4_MODEL" >&2; exit 1; }
        model_mnt="-v $GEMMA4_MODEL:/checkpoint:ro"; model_ref=/checkpoint ;;
    *)  local hf=${GEMMA4_HF_CACHE:-$HOME/.cache/huggingface}; mkdir -p "$hf"
        model_mnt="-v $hf:/root/.cache/huggingface"; model_ref="$GEMMA4_MODEL" ;;
  esac
  local tmpl_mnt="" tmpl_arg=""
  if [ -n "${GEMMA4_CHAT_TEMPLATE:-}" ]; then
    [ -f "$GEMMA4_CHAT_TEMPLATE" ] \
      || { echo "FAIL: chat template not found: $GEMMA4_CHAT_TEMPLATE" >&2; exit 1; }
    tmpl_mnt="-v $GEMMA4_CHAT_TEMPLATE:/chat-template.jinja:ro"
    tmpl_arg="--chat-template /chat-template.jinja"
  fi
  if [ -n "${GEMMA4_JIT_CACHE:-}" ]; then
    mkdir -p "$GEMMA4_JIT_CACHE"
    jit_mnt="-v $GEMMA4_JIT_CACHE:/jit-cache"
    jit_env="-e TRITON_CACHE_DIR=/jit-cache/triton -e TORCHINDUCTOR_CACHE_DIR=/jit-cache/inductor -e FLASHINFER_WORKSPACE_DIR=/jit-cache/flashinfer -e VLLM_CACHE_ROOT=/jit-cache/vllm"
  fi

  docker rm -f "$GEMMA4_CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$GEMMA4_CONTAINER" --network host --gpus all \
  ${GEMMA4_EXTRA_MOUNTS:-} \
    --shm-size=16g --ipc=host $model_mnt $mtp_mnt $jit_mnt $tmpl_mnt $jit_env \
    --entrypoint bash "$GEMMA4_IMAGE" -lc \
    "exec vllm serve $model_ref \
      --served-model-name $GEMMA4_SERVED_NAME \
      --host 0.0.0.0 --port $GEMMA4_PORT --tensor-parallel-size 1 \
      --max-model-len $GEMMA4_MAX_LEN --max-num-seqs $GEMMA4_MAX_SEQS \
      --max-num-batched-tokens $GEMMA4_BATCHED_TOKENS --enable-chunked-prefill \
      $prefix_cache_arg $lmo_arg --trust-remote-code \
      --reasoning-parser gemma4 --tool-call-parser gemma4 \
      --enable-auto-tool-choice \
      --limit-mm-per-prompt '{\"image\":0,\"audio\":0}' \
      --gpu-memory-utilization ${GEMMA4_GPU_UTIL:-0.50} $kv_arg $spec_arg $tmpl_arg $GEMMA4_EXTRA_ARGS"
  echo "started $GEMMA4_CONTAINER; waiting for :$GEMMA4_PORT (max ${GEMMA4_READY_S}s)"
  local i=0
  until api_ok; do
    i=$((i+15))
    [ "$(container_state)" = running ] \
      || { echo "FAIL: container exited" >&2; docker logs --tail 20 "$GEMMA4_CONTAINER" >&2 || true; exit 1; }
    [ $i -ge "$GEMMA4_READY_S" ] \
      && { echo "FAIL: not READY in ${GEMMA4_READY_S}s" >&2; docker logs --tail 20 "$GEMMA4_CONTAINER" >&2 || true; exit 1; }
    sleep 15
  done
  echo "READY (${i}s)"
  if [ "${GEMMA4_WARM_PREFILL:-1}" = 1 ]; then
    python3 - "$GEMMA4_PORT" "$GEMMA4_SERVED_NAME" <<'PY' || echo "WARN: startup warm prefill failed; server remains READY" >&2
import json
import sys
import time
import urllib.request

port, model = sys.argv[1:]
base = f"http://127.0.0.1:{port}"

def post(path, body, timeout=600):
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def token_ids(target, salt):
    words = [f"warmup-{salt}-{i:05d}-prefill-context" for i in range(target + 32)]
    text = " ".join(words)
    result = post("/tokenize", {"model": model, "prompt": text}, 120)
    ids = result.get("tokens")
    if not isinstance(ids, list) or len(ids) < target:
        raise RuntimeError(f"tokenizer returned fewer than {target} tokens")
    return ids[:target]

for label, count in (("short", 2048), ("chunk-boundary", 8192), ("long", 30000)):
    started = time.perf_counter()
    response = post("/v1/completions", {
        "model": model, "prompt": token_ids(count, label),
        "max_tokens": 1, "temperature": 0, "stream": False,
    })
    choices = response.get("choices") or []
    if not choices or choices[0].get("finish_reason") not in ("length", "stop"):
        raise RuntimeError(f"{label} warm request returned no completed choice")
    print(f"warm-prefill {label} tokens={count} elapsed={time.perf_counter()-started:.2f}s", flush=True)
PY
  fi
  free -g | awk 'NR==2 {printf "mem-GB total=%s used=%s available=%s\n", $2, $3, $7}'
  docker logs "$GEMMA4_CONTAINER" 2>&1 | grep -m1 'GPU KV cache size' \
    || echo "kv: 'GPU KV cache size' line not in container log"
}

cmd_down() {
  docker rm -f "$GEMMA4_CONTAINER"
  echo "down: $GEMMA4_CONTAINER removed"
}

cmd_status() {
  local st api log
  st=$(container_state)
  if api_ok; then api=OK; else api='---'; fi
  log="$(docker logs --tail 1 "$GEMMA4_CONTAINER" 2>&1 | tail -c 100 | tr '\n' ' ')" || true
  printf '%-14s %-9s api(:%s)=%-4s %s\n' "$GEMMA4_CONTAINER" "$st" "$GEMMA4_PORT" "$api" "$log"
}

cmd_smoke() {
  local t0 t1 resp
  t0=$(python3 -c 'import time;print(time.time())')
  resp="$(curl -s --max-time 300 "http://127.0.0.1:$GEMMA4_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$GEMMA4_SERVED_NAME"'","messages":[{"role":"user","content":"DGX Spark とは何か、日本語で一文で答えて。"}],"max_tokens":128,"temperature":0}')"
  t1=$(python3 -c 'import time;print(time.time())')
  RESP="$resp" python3 - "$t0" "$t1" <<'EOF'
import json, os, re, sys
d = json.loads(os.environ["RESP"])
msg = (d.get("choices") or [{}])[0].get("message") or {}
text = msg.get("content") or msg.get("reasoning") or ""
if not text:
    print("FAIL: empty body; raw=" + os.environ["RESP"][:500], file=sys.stderr); sys.exit(1)
if not re.search("[ぁ-ん]", text):
    print("FAIL: reply is not Japanese: " + text[:200], file=sys.stderr); sys.exit(1)
ct = (d.get("usage") or {}).get("completion_tokens") or 0
wall = float(sys.argv[2]) - float(sys.argv[1])
print(f"smoke: OK  {wall:.1f}s  completion_tokens={ct}  tok/s={ct/wall:.1f}")
print("body:", text[:300])
EOF
}

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  smoke) cmd_smoke ;;
  *) usage ;;
esac
