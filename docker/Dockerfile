FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy project definition and install dependencies
COPY pyproject.toml .
COPY fusesearch/ ./fusesearch/
ARG INSTALL_EXTRAS=all
RUN pip install --no-cache-dir ".[$INSTALL_EXTRAS]"

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "fusesearch", "serve"]
