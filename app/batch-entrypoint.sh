#!/usr/bin/env bash
# Docker 専用バッチ実行スクリプト
# 環境変数は Docker Compose の env_file / environment で注入済みのため、
# .env ファイルのソースは行わない（DB ホストなどが上書きされるのを防ぐため）。

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
SCRIPT_DIR="${PROJECT_ROOT}/batch/scripts"
APP_ENV="${APP_ENV:-development}"
BATCH_DATA_DIR="${BATCH_DATA_DIR:-${PROJECT_ROOT}/data/${APP_ENV}}"
COMMENTS_DIR="${COMMENTS_DIR:-${BATCH_DATA_DIR}/comments}"
COMMUNITY_NOTE_BACKUP_DIR="${COMMUNITY_NOTE_BACKUP_DIR:-${BATCH_DATA_DIR}/oldcommunitylog}"
LOG_DIR="${LOG_DIR:-${BATCH_DATA_DIR}/logs}"

if [ -z "${TARGET_USERS_CSV:-}" ]; then
    if [ -f "${BATCH_DATA_DIR}/targetusers.csv" ]; then
        TARGET_USERS_CSV="${BATCH_DATA_DIR}/targetusers.csv"
    else
        TARGET_USERS_CSV="${PROJECT_ROOT}/targetusers.csv"
    fi
fi

mkdir -p "${BATCH_DATA_DIR}" "${COMMENTS_DIR}" "${COMMUNITY_NOTE_BACKUP_DIR}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/batch_${TIMESTAMP}.log"
START_TS="$(date +%s)"

export PROJECT_ROOT APP_ENV BATCH_DATA_DIR TARGET_USERS_CSV VODS_CSV COMMENTS_DIR COMMUNITY_NOTE_BACKUP_DIR

VODS_CSV="${VODS_CSV:-${BATCH_DATA_DIR}/batch_twitch_vods_all.csv}"
export VODS_CSV

run_required() {
    local name="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${name}..." | tee -a "${LOG_FILE}"
    if ! python "${SCRIPT_DIR}/${name}" >> "${LOG_FILE}" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error in ${name}" | tee -a "${LOG_FILE}"
        exit 1
    fi
}

run_optional() {
    local name="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${name}..." | tee -a "${LOG_FILE}"
    if ! python "${SCRIPT_DIR}/${name}" >> "${LOG_FILE}" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: ${name} failed (non-fatal)" | tee -a "${LOG_FILE}"
    fi
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting batch process" | tee -a "${LOG_FILE}"

run_required "get_vod_list_batch.py"
run_required "batch_download_comments.py"
run_required "insertdb.py"

if [ "${SKIP_FAISS:-0}" != "1" ] && [ -n "${FAISS_API_URL:-}" ]; then
    run_optional "build_faiss_index.py"
elif [ "${SKIP_FAISS:-0}" = "1" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] build_faiss_index.py をスキップ (SKIP_FAISS=1)" | tee -a "${LOG_FILE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] build_faiss_index.py をスキップ (FAISS_API_URL 未設定)" | tee -a "${LOG_FILE}"
fi

run_optional "generate_community_notes.py"

END_TS="$(date +%s)"
ELAPSED="$((END_TS - START_TS))"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch completed successfully. Elapsed: ${ELAPSED}s" | tee -a "${LOG_FILE}"
