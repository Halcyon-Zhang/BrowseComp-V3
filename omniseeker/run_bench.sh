#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNISEEKER_ROOT="${SCRIPT_DIR}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${OMNISEEKER_ROOT}/logs"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${OMNISEEKER_ROOT}/src:${PYTHONPATH:-}"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

PROVIDER="${PROVIDER:-custom}"
MODEL_NAME="${MODEL_NAME:-}"
ENV_MODE="${ENV_MODE:-http_mcp_search}"
RESULT_DIR="${RESULT_DIR:-}"
DATASET="${DATASET:-Life}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/samples}"
DATA_DIR="${DATA_DIR:-}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
MAX_TURNS="${MAX_TURNS:-20}"
PORT="${PORT:-9191}"
CONFIG="${CONFIG:-${OMNISEEKER_ROOT}/configs/gateway_searchtools.example.json}"
MCP_SERVER_URL="${MCP_SERVER_URL:-}"
START_MCP="${START_MCP:-auto}"   # auto|on|off
EXAMPLE="${EXAMPLE:-}"
DRY_RUN="${DRY_RUN:-0}"
SHOW_CMD="${SHOW_CMD:-1}"
OPENAI_API_BASE_OVERRIDE=""
OPENAI_API_KEY_OVERRIDE=""
USER_SET_PORT=0

usage() {
  cat <<'EOF'
Usage:
  bash omniseeker/run_bench.sh [options]

Core options:
  --provider NAME         Provider preset: custom|gpt|kimi|doubao|openrouter|mimo
  --model-name NAME       Model name passed to rollout
  --env-mode MODE         http_mcp_search | tool_free (default: http_mcp_search)
  --dataset NAME          Dataset subdir under data root (default: Life)
  --data-root PATH        Root containing datasets (default: data/samples)
  --data-dir PATH         Absolute dataset dir (overrides --data-root + --dataset)
  --output-dir PATH       Absolute output dir (default derives from --env-mode)
  --num-rollouts N        Parallel rollout workers (default: 2)
  --max-turns N           Max turns per sample (default: 20)
  --example N             Optional sample cap: first N items

MCP options:
  --port N                Local MCP server port when starting one (default: 9191)
  --gateway-config PATH   Gateway config file (default: omniseeker/configs/gateway_searchtools.example.json)
  --mcp-server-url URL    Existing MCP server url (default: http://localhost:<port>)
  --start-mcp MODE        auto|on|off (default: auto)

Other:
  --openai-api-base URL   Override OPENAI_API_BASE for this run
  --openai-api-key KEY    Override OPENAI_API_KEY for this run
  --dry-run               Print commands without executing
  -h, --help              Show this help

Credential notes:
  OPENAI_API_KEY must be provided via shell/.env or --openai-api-key.
  OPENAI_API_BASE is inferred from provider preset if not explicitly set.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:-}"; shift 2 ;;
    --env-mode) ENV_MODE="${2:-}"; shift 2 ;;
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --data-dir) DATA_DIR="${2:-}"; shift 2 ;;
    --output-dir) RESULT_DIR="${2:-}"; shift 2 ;;
    --num-rollouts) NUM_ROLLOUTS="${2:-}"; shift 2 ;;
    --max-turns) MAX_TURNS="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; USER_SET_PORT=1; shift 2 ;;
    --gateway-config) CONFIG="${2:-}"; shift 2 ;;
    --mcp-server-url) MCP_SERVER_URL="${2:-}"; shift 2 ;;
    --start-mcp) START_MCP="${2:-}"; shift 2 ;;
    --example) EXAMPLE="${2:-}"; shift 2 ;;
    --openai-api-base) OPENAI_API_BASE_OVERRIDE="${2:-}"; shift 2 ;;
    --openai-api-key) OPENAI_API_KEY_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

apply_provider_preset() {
  case "${PROVIDER}" in
    custom)
      MODEL_NAME="${MODEL_NAME:-gpt-5-mini}"
      ;;
    gpt)
      MODEL_NAME="${MODEL_NAME:-gpt-5-mini}"
      if [[ "${USER_SET_PORT}" -eq 0 ]]; then
        PORT=9191
      fi
      ;;
    kimi)
      MODEL_NAME="${MODEL_NAME:-kimi-k2.5}"
      if [[ "${USER_SET_PORT}" -eq 0 ]]; then
        PORT=9191
      fi
      if [[ -z "${OPENAI_API_BASE_OVERRIDE}" && -z "${OPENAI_API_BASE:-}" ]]; then
        OPENAI_API_BASE_OVERRIDE="https://api.moonshot.cn/v1"
      fi
      ;;
    doubao)
      MODEL_NAME="${MODEL_NAME:-doubao-seed-1-8-251228}"
      if [[ "${USER_SET_PORT}" -eq 0 ]]; then
        PORT=9192
      fi
      if [[ -z "${OPENAI_API_BASE_OVERRIDE}" && -z "${OPENAI_API_BASE:-}" ]]; then
        OPENAI_API_BASE_OVERRIDE="https://ark.cn-beijing.volces.com/api/v3"
      fi
      ;;
    openrouter)
      MODEL_NAME="${MODEL_NAME:-google/gemini-3-flash-preview}"
      if [[ "${USER_SET_PORT}" -eq 0 ]]; then
        PORT=9194
      fi
      if [[ -z "${OPENAI_API_BASE_OVERRIDE}" && -z "${OPENAI_API_BASE:-}" ]]; then
        OPENAI_API_BASE_OVERRIDE="https://openrouter.ai/api/v1"
      fi
      ;;
    mimo)
      MODEL_NAME="${MODEL_NAME:-mimo-v2-flash}"
      if [[ "${USER_SET_PORT}" -eq 0 ]]; then
        PORT=9193
      fi
      if [[ -z "${OPENAI_API_BASE_OVERRIDE}" && -z "${OPENAI_API_BASE:-}" ]]; then
        OPENAI_API_BASE_OVERRIDE="https://api.xiaomimimo.com/v1"
      fi
      ;;
    *)
      echo "[ERROR] Unknown provider preset: ${PROVIDER}" >&2
      echo "[ERROR] Valid values: custom|gpt|kimi|doubao|openrouter|mimo" >&2
      exit 1
      ;;
  esac
}

apply_provider_preset

if [[ -z "${RESULT_DIR}" ]]; then
  if [[ "${ENV_MODE}" == "tool_free" ]]; then
    RESULT_DIR="${PROJECT_ROOT}/results_no_tool"
  else
    RESULT_DIR="${PROJECT_ROOT}/results"
  fi
fi

if [[ -z "${DATA_DIR}" ]]; then
  DATA_DIR="${DATA_ROOT}/${DATASET}"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] gateway config not found: ${CONFIG}" >&2
    exit 1
  fi
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] dataset directory not found: ${DATA_DIR}" >&2
    exit 1
  fi
fi

if [[ -z "${MCP_SERVER_URL}" ]]; then
  MCP_SERVER_URL="http://localhost:${PORT}"
fi

case "${START_MCP}" in
  auto|on|off) ;;
  *) echo "[ERROR] --start-mcp must be one of auto|on|off" >&2; exit 1 ;;
esac

start_local_mcp=0
if [[ "${START_MCP}" == "on" ]]; then
  start_local_mcp=1
elif [[ "${START_MCP}" == "off" ]]; then
  start_local_mcp=0
elif [[ "${ENV_MODE}" == "tool_free" ]]; then
  start_local_mcp=0
elif [[ "${MCP_SERVER_URL}" == "http://localhost:${PORT}" || "${MCP_SERVER_URL}" == "http://127.0.0.1:${PORT}" ]]; then
  start_local_mcp=1
fi

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_TAG="$(echo "${PROVIDER}_${MODEL_NAME}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
SERVER_LOG="${LOG_DIR}/mcpserver_${LOG_TAG}_${PORT}_${TS}.log"
ROLL_LOG="${LOG_DIR}/rollout_${LOG_TAG}_${TS}.log"
PID_FILE="${LOG_DIR}/mcpserver_${PORT}.pid"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping MCP server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_port_ready() {
  local port="$1"
  for _ in $(seq 1 30); do
    if python3 - "$port" <<'PY'
import socket
import sys
s = socket.socket()
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

cd "${PROJECT_ROOT}"

DEFAULT_CACHE_ROOT="${PROJECT_ROOT}/.cache"
export MMDEEPSEARCH_CACHE_DB="${MMDEEPSEARCH_CACHE_DB:-${DEFAULT_CACHE_ROOT}/search_cache.db}"
export SEARCH_LOG_DIR="${SEARCH_LOG_DIR:-${OMNISEEKER_ROOT}/logs}"
export IMAGE_DOWNLOAD_LOG_DIR="${IMAGE_DOWNLOAD_LOG_DIR:-${OMNISEEKER_ROOT}/logs/image_downloads}"
mkdir -p "$(dirname "${MMDEEPSEARCH_CACHE_DB}")"

if [[ -n "${OPENAI_API_BASE_OVERRIDE}" ]]; then
  export OPENAI_API_BASE="${OPENAI_API_BASE_OVERRIDE}"
fi
if [[ -n "${OPENAI_API_KEY_OVERRIDE}" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY_OVERRIDE}"
fi

if [[ "${DRY_RUN}" -eq 0 && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set. Provide it via environment or --openai-api-key." >&2
  exit 1
fi

if [[ "${start_local_mcp}" -eq 1 ]]; then
  echo "[INFO] Starting local MCP server at ${MCP_SERVER_URL}"
  if command -v lsof >/dev/null 2>&1 && lsof -i :"${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    PIDS="$(lsof -i :"${PORT}" -sTCP:LISTEN -t)"
    echo "[INFO] Killing existing process(es) on port ${PORT}: ${PIDS}"
    kill ${PIDS} >/dev/null 2>&1 || true
    sleep 1
  fi
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    python3 "${OMNISEEKER_ROOT}/src/mcp_server/main.py" \
      --config "${CONFIG}" \
      --port "${PORT}" \
      > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "${SERVER_PID}" > "${PID_FILE}"
    if ! wait_port_ready "${PORT}"; then
      echo "[ERROR] MCP server failed to become ready on port ${PORT}" >&2
      exit 1
    fi
  fi
fi

cmd=(
  python3 "${OMNISEEKER_ROOT}/src/run_rollout.py"
  --data_dir "${DATA_DIR}"
  --env_mode "${ENV_MODE}"
  --num_rollouts "${NUM_ROLLOUTS}"
  --model_name "${MODEL_NAME}"
  --mcp_server_url "${MCP_SERVER_URL}"
  --gateway_config_path "${CONFIG}"
  --output_dir "${RESULT_DIR}"
  --max_turns "${MAX_TURNS}"
  --dataset "${DATASET}"
)
if [[ -n "${EXAMPLE}" ]]; then
  cmd+=(--example "${EXAMPLE}")
fi

echo "[INFO] =========================================="
echo "[INFO] Unified benchmark runner"
echo "[INFO] provider=${PROVIDER}"
echo "[INFO] model=${MODEL_NAME}"
echo "[INFO] OPENAI_API_BASE=${OPENAI_API_BASE:-<default>}"
echo "[INFO] env_mode=${ENV_MODE}"
echo "[INFO] dataset=${DATASET}"
echo "[INFO] data_dir=${DATA_DIR}"
echo "[INFO] output_dir=${RESULT_DIR}"
echo "[INFO] mcp_url=${MCP_SERVER_URL} (start_local_mcp=${start_local_mcp})"
echo "[INFO] log=${ROLL_LOG}"
echo "[INFO] =========================================="

if [[ "${SHOW_CMD}" -eq 1 ]]; then
  printf '[INFO] command:'
  printf ' %q' "${cmd[@]}"
  echo
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] done"
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "${ROLL_LOG}"
