#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/batch/scripts/insertdb.py"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

ENV_FILE_ARG=""
COMMENTS_DIR_ARG=""
REINGEST_EXISTING_VODS=0

usage() {
    cat <<'EOF'
Usage: ./run_import_comments.sh [--env-file PATH] [--comments-dir PATH] [--reingest-existing-vods]

Options:
  --env-file PATH            Load environment variables from PATH before import.
  --comments-dir PATH        Directory that contains <vod_id>.json files.
  --reingest-existing-vods   Re-process VODs even if they already exist in DB.
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
        --comments-dir)
            if [ -z "${2:-}" ]; then
                echo "Error: --comments-dir requires a path" >&2
                exit 1
            fi
            COMMENTS_DIR_ARG="$2"
            shift 2
            ;;
        --reingest-existing-vods)
            REINGEST_EXISTING_VODS=1
            shift
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
COMMENTS_DIR="${COMMENTS_DIR_ARG:-${COMMENTS_DIR:-${ROOT_DIR}/data/${APP_ENV}/comments}}"
if [ "${COMMENTS_DIR#/}" = "${COMMENTS_DIR}" ]; then
    COMMENTS_DIR="${ROOT_DIR}/${COMMENTS_DIR}"
fi

if [ ! -d "${COMMENTS_DIR}" ]; then
    echo "Error: comments directory not found: ${COMMENTS_DIR}" >&2
    exit 1
fi

export PROJECT_ROOT="${ROOT_DIR}"
export ENV_FILE
export COMMENTS_DIR

CMD=("${PYTHON_BIN}" "${SCRIPT_PATH}" "--comments-dir" "${COMMENTS_DIR}")
if [ "${REINGEST_EXISTING_VODS}" = "1" ]; then
    CMD+=("--reingest-existing-vods")
fi

echo "Importing comments from: ${COMMENTS_DIR}"
echo "Using env file: ${ENV_FILE}"
"${CMD[@]}"
