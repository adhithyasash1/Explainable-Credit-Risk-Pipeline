# --- Stage 1: Builder ---
FROM python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- Stage 2: Final Image ---
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app

# Copy and install Python packages
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application code first (as root to ensure proper ownership)
COPY --chown=app:app ./app ./app

# Create a directory for model artifacts if they need to be mounted
RUN mkdir -p /app/models && chown -R app:app /app

# Switch to non-root user
USER app

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
