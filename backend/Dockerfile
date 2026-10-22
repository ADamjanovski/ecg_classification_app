# backend/Dockerfile
FROM python:3.12.3-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --frozen --no-dev \
    --index-url https://download.pytorch.org/whl/cpu && \
    find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete && \
    find /app/.venv -type f -name "*.pyo" -delete && \
    rm -rf /root/.cache/uv

FROM python:3.12.3-slim-bookworm AS production

WORKDIR /app

COPY --from=builder /app/.venv .venv

COPY app ./app
COPY src ./src
COPY CNN1d.py ./CNN1d.py

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--log-level", "info", "--host", "0.0.0.0", "--port", "8000"]