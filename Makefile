-include .env
export

.PHONY: help \
        build up start stop down restart logs status clean lint fmt tests \
        index index-confluence search ask mcp \
        example-basic example-ask example-custom example-clean \
        package publish release

include examples/examples.mk

## help    : Print commands help.
help : Makefile examples/examples.mk
	@sed -n 's/^## *//p' $^ | tr -s '\t' ' ' | column -t -s ':'

## build   : Build Docker images.
build:
	docker compose build

## up      : Start services (attached).
up:
	docker compose up --remove-orphans

## start   : Start all services (detached).
start:
	docker compose up -d --remove-orphans

## stop    : Stop all services.
stop down:
	docker compose down --remove-orphans

## restart : Restart all services.
restart: stop start

## logs    : Tail logs from all services.
logs:
	docker compose logs -f

## status  : Show running services.
status:
	docker compose ps

## index   : Index sample docs (run after 'make start').
index:
	docker compose exec fusesearch python -m fusesearch index /app/data/docs

## index-confluence : Index from Confluence (run after 'make start', needs .env).
index-confluence:
	docker compose exec fusesearch python -m fusesearch index --source confluence

## search  : Search indexed docs. Usage: make search "your query"
search:
	docker compose exec fusesearch python -m fusesearch search "$(filter-out $@,$(MAKECMDGOALS))"

## ask     : Ask a question. Usage: make ask "your question"
ask:
	docker compose exec fusesearch python -m fusesearch ask "$(filter-out $@,$(MAKECMDGOALS))"

%:
	@:

## mcp     : Show MCP server URL for Claude Code config.
mcp:
	@echo "MCP server running at http://localhost:$${MCP_PORT:-8001}/mcp"
	@echo ""
	@echo "Add to Claude Code:  claude mcp add fusesearch http://localhost:$${MCP_PORT:-8001}/mcp --transport http"

## lint    : Run ruff linter and formatter check.
lint:
	docker compose run --rm --no-deps \
		-v ./fusesearch:/app/fusesearch \
		-v ./tests:/app/tests \
		fusesearch ruff check fusesearch/ tests/
	docker compose run --rm --no-deps \
		-v ./fusesearch:/app/fusesearch \
		-v ./tests:/app/tests \
		fusesearch ruff format --check fusesearch/ tests/

## tests   : Run tests.
tests:
	docker compose run --rm --no-deps \
		-v ./tests:/app/tests \
		fusesearch pytest

## fmt     : Auto-format code with ruff.
fmt:
	docker compose run --rm --no-deps \
		-v ./fusesearch:/app/fusesearch \
		-v ./tests:/app/tests \
		fusesearch ruff format fusesearch/ tests/

## package : Build Python package (wheel + sdist) into dist/.
package:
	docker build -f docker/build.Dockerfile -t fusesearch-build .
	docker run --rm -v ./dist:/out fusesearch-build

## publish : Publish package to PyPI (requires PYPI_TOKEN in .env).
publish: package
	docker run --rm fusesearch-build twine upload /app/dist/* -u __token__ -p $${PYPI_TOKEN}

## release : Create a GitHub release from the version in pyproject.toml.
release:
	$(eval VERSION := $(shell grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/'))
	gh release create v$(VERSION) --generate-notes --title "v$(VERSION)"

## clean   : Stop services and remove volumes.
clean:
	docker compose down -v --remove-orphans
