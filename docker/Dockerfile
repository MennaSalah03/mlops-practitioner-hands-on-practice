FROM python:3.12-slim AS builder
WORKDIR /build

# Build dependencies in an isolated layer
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 appuser

# Only bring over the compiled libraries from the builder
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser models/ /app/models/

WORKDIR /app

ENV PYTHONPATH=/app/src

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

# 3. Run uvicorn as a Python module rather than a standalone script
CMD ["python", "-m", "uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
