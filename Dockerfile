FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
RUN useradd --create-home sentinel && mkdir -p /app/state && chown sentinel /app/state
USER sentinel

# Ollama runs on the host: on Linux add `--add-host=host.docker.internal:host-gateway`.
ENV SENTINEL_OLLAMA_URL=http://host.docker.internal:11434 \
    SENTINEL_DB_PATH=/app/state/sentinel.db

EXPOSE 8000
CMD ["sentinel", "serve", "--host", "0.0.0.0", "--port", "8000"]
