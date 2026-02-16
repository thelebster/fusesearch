FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir build twine

COPY pyproject.toml README.md LICENSE ./
COPY fusesearch/ ./fusesearch/

RUN python -m build

CMD ["sh", "-c", "cp dist/* /out/"]
