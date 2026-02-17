FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir build twine

COPY pyproject.toml README.md LICENSE ./
COPY fusesearch/ ./fusesearch/

# Convert relative image paths to absolute GitHub URLs for PyPI
RUN sed -i -E 's|!\[([^]]*)\]\(docs/|![\1](https://raw.githubusercontent.com/thelebster/fusesearch/master/docs/|g' README.md

RUN python -m build

CMD ["sh", "-c", "cp dist/* /out/"]
