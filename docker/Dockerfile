FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Pre-install torch from a chosen index, in its own layer (kept before the
# pyproject.toml COPY so dependency changes don't re-download it).
#   default (CPU wheel, a few hundred MB): https://download.pytorch.org/whl/cpu
#   NVIDIA GPU host: a CUDA index from https://pytorch.org/get-started/locally/
#   empty: skip; pip resolves torch from PyPI (CUDA build, several GB) if an
#          extra needs it, or not at all when INSTALL_EXTRAS has no "local"
# Declared here, not at the top, so changing INSTALL_EXTRAS doesn't rebuild this layer.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir --timeout=120 torch --index-url "$TORCH_INDEX_URL"; \
    fi

# Install dependencies
COPY pyproject.toml .
COPY fusesearch/__init__.py ./fusesearch/
ARG INSTALL_EXTRAS=all
RUN pip install --no-cache-dir --timeout=120 ".[$INSTALL_EXTRAS]"

# Copy source code (PYTHONPATH ensures /app/fusesearch/ is found before
# the stub package pip installed to site-packages during dependency resolution)
COPY fusesearch/ ./fusesearch/
ENV PYTHONPATH=/app

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "fusesearch", "serve"]
