# EcoRoute SDN Controller Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Build React frontend
FROM node:18-alpine as frontend-builder

WORKDIR /app/dashboard/frontend

COPY dashboard/frontend/package*.json ./
RUN npm ci --only=production

COPY dashboard/frontend/ ./
RUN npm run build

# Stage 2: Python application
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash ecoroute

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=ecoroute:ecoroute . .

# Copy built frontend
COPY --from=frontend-builder /app/dashboard/frontend/build /app/dashboard/frontend/build

# Create logs directory
RUN mkdir -p /app/logs && chown -R ecoroute:ecoroute /app/logs

# Switch to non-root user
USER ecoroute

# Expose ports
# 6653 - OpenFlow controller
# 5000 - Dashboard API
# 8080 - Controller REST API
EXPOSE 6653 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Default command - run controller
CMD ["python", "-m", "controller.ecoroute_controller"]
