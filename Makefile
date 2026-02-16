-include .env
export

.PHONY: help \
        build up start stop down restart logs status clean lint fmt \
        index search mcp

## help    : Print commands help.
help : Makefile
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

## search  : Search indexed docs. Usage: make search "your query"
search:
	docker compose exec fusesearch python -m fusesearch search "$(filter-out $@,$(MAKECMDGOALS))"

%:
	@:

## mcp     : Show MCP server URL for Claude Code config.
mcp:
	@echo "MCP server running at http://localhost:$${MCP_PORT:-8001}/sse"
	@echo ""
	@echo "Add to Claude Code:  claude mcp add fusesearch http://localhost:$${MCP_PORT:-8001}/sse --transport sse"

## lint    : Run ruff linter and formatter check.
lint:
	docker compose run --rm fusesearch ruff check fusesearch/
	docker compose run --rm fusesearch ruff format --check fusesearch/

## fmt     : Auto-format code with ruff.
fmt:
	docker compose run --rm fusesearch ruff format fusesearch/

## clean   : Stop services and remove volumes.
clean:
	docker compose down -v --remove-orphans
