#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/batch/scripts"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
ENV_FILE_ARG=""

usage() {
    cat <<'EOF'
Usage: ./run_batch.sh [--env-file PATH]

Options:
  --env-file PATH   Load environment variables from PATH before running batch scripts.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env-file)
            if [ -z "${2:-}" ]; then
                echo "Error: --env-file requires a path" >&2
                exit 1
            fi
            ENV_FILE_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

ENV_FILE="${ENV_FILE_ARG:-${ENV_FILE:-${ROOT_DIR}/.env}}"
if [ "${ENV_FILE#/}" = "${ENV_FILE}" ]; then
    ENV_FILE="${ROOT_DIR}/${ENV_FILE}"
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: env file not found: ${ENV_FILE}" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
. "${ENV_FILE}"
set +a

APP_ENV="${APP_ENV:-development}"
BATCH_DATA_DIR="${BATCH_DATA_DIR:-${ROOT_DIR}/data/${APP_ENV}}"
VODS_CSV="${VODS_CSV:-${BATCH_DATA_DIR}/batch_twitch_vods_all.csv}"
COMMENTS_DIR="${COMMENTS_DIR:-${BATCH_DATA_DIR}/comments}"
COMMUNITY_NOTE_BACKUP_DIR="${COMMUNITY_NOTE_BACKUP_DIR:-${BATCH_DATA_DIR}/oldcommunitylog}"
LOG_DIR="${LOG_DIR:-${BATCH_DATA_DIR}/logs}"

if [ -z "${TARGET_USERS_CSV:-}" ]; then
    if [ -f "${BATCH_DATA_DIR}/targetusers.csv" ]; then
        TARGET_USERS_CSV="${BATCH_DATA_DIR}/targetusers.csv"
    else
        TARGET_USERS_CSV="${ROOT_DIR}/targetusers.csv"
    fi
fi

mkdir -p "${BATCH_DATA_DIR}" "${COMMENTS_DIR}" "${COMMUNITY_NOTE_BACKUP_DIR}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/batch_${TIMESTAMP}.log"
START_TS="$(date +%s)"

export PROJECT_ROOT="${ROOT_DIR}"
export ENV_FILE
export APP_ENV
export BATCH_DATA_DIR
export TARGET_USERS_CSV
export VODS_CSV
export COMMENTS_DIR
export COMMUNITY_NOTE_BACKUP_DIR

run_required() {
    local script_name="$1"
    echo "Running ${script_name}..." >> "${LOG_FILE}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/${script_name}" >> "${LOG_FILE}" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error in ${script_name}" >> "${LOG_FILE}"
        exit 1
    fi
}

run_optional() {
    local script_name="$1"
    echo "Running ${script_name}..." >> "${LOG_FILE}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/${script_name}" >> "${LOG_FILE}" 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: ${script_name} failed (non-fatal)" >> "${LOG_FILE}"
    fi
}

echo "Starting batch process at $(date)" >> "${LOG_FILE}"

run_required "get_vod_list_batch.py"
run_required "batch_download_comments.py"
run_required "insertdb.py"

if [ "${SKIP_FAISS:-0}" != "1" ]; then
    run_optional "build_faiss_index.py"
fi

run_optional "generate_community_notes.py"

END_TS="$(date +%s)"
ELAPSED="$((END_TS - START_TS))"

echo "Batch process completed successfully at $(date)" >> "${LOG_FILE}"
echo "Total elapsed time: ${ELAPSED} sec" >> "${LOG_FILE}"
