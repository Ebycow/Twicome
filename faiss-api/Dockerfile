# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# libgomp1: faiss-cpu が要求する OpenMP ランタイム
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 埋め込みモデルを事前ダウンロード（起動時のDL待ちを回避）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('hotchpotch/static-embedding-japanese', device='cpu')"

COPY main.py .

EXPOSE 8100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
