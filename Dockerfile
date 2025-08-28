# --- Stage 1: Builder ---
# This stage installs all dependencies, including build-time dependencies.
FROM python:3.9-slim AS builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- Stage 2: Final Image ---
# This stage creates the final, lean production image.
FROM python:3.9-slim

WORKDIR /app

# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app
USER app

# Copy installed wheels from the builder stage and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the application code and model artifacts
COPY --chown=app:app ./app ./app

# Expose the port the app runs on
EXPOSE 8000

# Add a health check to ensure the container is running correctly
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application with Gunicorn
# Using 4 workers is a common starting point. Adjust based on performance testing.
# The UvicornWorker class is used to run the ASGI application.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
