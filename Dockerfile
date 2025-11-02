# Multi-stage Dockerfile for Kaine AI Production Deployment
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements-web.txt requirements-prod.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-web.txt -r requirements-prod.txt

# Stage 2: Runtime - Create minimal production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY kaine_ai.py web_app.py ./
COPY static/ ./static/
COPY data/ ./data/

# Create cache directory with proper permissions
RUN mkdir -p cache && chmod 755 cache

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash kaineuser && \
    chown -R kaineuser:kaineuser /app

# Switch to non-root user
USER kaineuser

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "web_app.py"]
