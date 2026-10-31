EXAMPLES_COMPOSE = docker compose -f examples/docker-compose.yml

## example-basic  : Run basic_search example (isolated Qdrant).
example-basic:
	$(EXAMPLES_COMPOSE) run --rm basic_search

## example-ask    : Run ask example (isolated Qdrant).
example-ask:
	$(EXAMPLES_COMPOSE) run --rm ask

## example-custom : Run custom_source example (isolated Qdrant).
example-custom:
	$(EXAMPLES_COMPOSE) run --rm custom_source

## example-multi-tenant : Run multi_tenant example (isolated Qdrant).
example-multi-tenant:
	$(EXAMPLES_COMPOSE) run --rm multi_tenant

## example-clean  : Remove example containers and volumes.
example-clean:
	$(EXAMPLES_COMPOSE) down -v
