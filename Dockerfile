# --- Build Stage ---
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies (if any)
# For this project, we can just install everything here

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# --- Final Stage ---
FROM python:3.9-slim

WORKDIR /app

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY --from=builder /app/app .

# Expose the port and run the app with Gunicorn for production
EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]
