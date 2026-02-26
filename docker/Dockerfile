FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

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
