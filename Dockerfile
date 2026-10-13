# Objective 3: flight price API image (multi-stage, non-root, healthchecked)
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY api/ api/
# Models are baked in so the image is self-contained and versioned with its tag.
COPY models/ models/
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["gunicorn", "-c", "api/gunicorn.conf.py", "api.app:app"]
