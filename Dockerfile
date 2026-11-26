FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only torch keeps the image ~4x smaller than the CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/lstm_autoencoder.pt artifacts/metrics.json artifacts/
COPY reports/ reports/
COPY tests/fixtures/ tests/fixtures/
COPY dashboard/ dashboard/

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "pdm.api:app", "--host", "0.0.0.0", "--port", "8000"]
