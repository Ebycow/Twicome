FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 埋め込みモデルを事前ダウンロード（起動時のDL待ちを回避）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('hotchpotch/static-embedding-japanese', device='cpu')"

COPY main.py .
COPY app_factory.py .
COPY faiss_search.py .
COPY core/ ./core/
COPY services/ ./services/
COPY routers/ ./routers/
COPY templates/ ./templates/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
