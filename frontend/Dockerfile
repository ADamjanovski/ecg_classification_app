FROM python:3.11.15-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only the dependency files first so this layer is cached
# and not re-run unless pyproject.toml or uv.lock actually change.
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# ── Production stage ──────────────────────────────────────────────────────────

FROM python:3.11.15-slim AS production

WORKDIR /app

# Create non-root user and give them ownership of /app before switching.
RUN adduser --disabled-password --gecos "" appuser && chown appuser /app

# Copy only the built virtualenv from the builder; no uv, no build cache.
COPY --from=builder /app/.venv /app/.venv

# Copy application files with correct ownership.
COPY --chown=appuser app.py .
COPY --chown=appuser .streamlit/ .streamlit/

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
