# Blender AI Agent Studio Production Dockerfile
FROM ubuntu:22.04 AS base

# Build arguments
ARG BLENDER_VERSION=4.0
ARG PYTHON_VERSION=3.11

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BLENDER_VERSION=${BLENDER_VERSION} \
    PYTHON_VERSION=${PYTHON_VERSION} \
    DISPLAY=:99 \
    PATH="/opt/blender:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-dev \
    python3-venv \
    build-essential \
    # Blender dependencies
    wget \
    curl \
    xvfb \
    xauth \
    libxi6 \
    libxrender1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxkbcommon-x11-0 \
    libxss1 \
    libasound2 \
    libpulse0 \
    # Additional utilities
    git \
    ca-certificates \
    software-properties-common \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r blender && useradd -r -g blender -d /app -s /bin/bash blender

# Install Blender
RUN wget -q https://download.blender.org/release/Blender${BLENDER_VERSION}/blender-${BLENDER_VERSION}.0-linux-x64.tar.xz \
    && tar -xf blender-${BLENDER_VERSION}.0-linux-x64.tar.xz \
    && mv blender-${BLENDER_VERSION}.0-linux-x64 /opt/blender \
    && rm blender-${BLENDER_VERSION}.0-linux-x64.tar.xz \
    && chown -R blender:blender /opt/blender

# Development stage
FROM base AS development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    # Development tools
    vim \
    nano \
    htop \
    tree \
    # Testing tools
    firefox-esr \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt* ./

# Create virtual environment and install dependencies
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi

# Copy application code
COPY --chown=blender:blender . .

# Create necessary directories
RUN mkdir -p output logs cache assets temp \
    && chown -R blender:blender /app

# Switch to non-root user
USER blender

# Production stage
FROM base AS production

# Set working directory
WORKDIR /app

# Copy requirements and install production dependencies only
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache
    pip cache purge

# Copy application code
COPY --chown=blender:blender . .

# Create necessary directories with proper permissions
RUN mkdir -p output logs cache assets temp \
    && chown -R blender:blender /app \
    && chmod 755 /app/output /app/logs /app/cache /app/assets /app/temp

# Create entrypoint script with enhanced features
COPY <<EOF /entrypoint.sh
#!/bin/bash
set -e

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "\$XVFB_PID" ]; then
        kill \$XVFB_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start virtual display
echo "Starting virtual display..."
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
XVFB_PID=\$!

# Wait for display to start
sleep 2

# Verify Blender installation
echo "Verifying Blender installation..."
/opt/blender/blender --version

# Run application health check if available
if [ -f "health_check.py" ]; then
    echo "Running health check..."
    python health_check.py
fi

# Execute main command
echo "Starting application: \$@"
exec "\$@"
EOF

RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER blender

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501')" || exit 1

# Expose ports
EXPOSE 8501 8000

# Set default environment variables
ENV BLENDER_PATH=/opt/blender/blender \
    GOOGLE_API_KEY="" \
    OUTPUT_DIRECTORY=/app/output \
    LOG_LEVEL=INFO \
    CACHE_SIZE_MB=512 \
    ENABLE_CACHING=true

ENTRYPOINT ["/entrypoint.sh"]

# Default command - run web interface
CMD ["streamlit", "run", "web_interface/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# GPU-enabled stage
FROM production AS gpu

USER root

# Install NVIDIA container runtime dependencies
RUN apt-get update && apt-get install -y \
    nvidia-container-runtime \
    && rm -rf /var/lib/apt/lists/*

# Configure Blender for GPU rendering
ENV CYCLES_DEVICE=CUDA

USER blender

# Multi-architecture build support
FROM production AS final

# Add build information
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="Blender AI Agent Studio" \
      org.label-schema.description="AI-powered 3D scene generation platform" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-repo/blender-ai-agent-studio" \
      org.label-schema.schema-version="1.0" \
      maintainer="Blender AI Agent Studio Team"