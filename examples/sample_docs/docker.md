# Docker

Docker is a platform for developing, shipping, and running applications inside containers. Containers package an application with all its dependencies, ensuring it runs consistently across different environments.

## Containers vs Virtual Machines

Unlike virtual machines, containers share the host OS kernel and do not require a full operating system per instance. This makes them lightweight — a container can start in seconds and uses a fraction of the memory a VM would need.

## Docker Compose

Docker Compose is a tool for defining and running multi-container applications. A `docker-compose.yml` file describes services, networks, and volumes. With a single `docker compose up` command, you can spin up an entire application stack.

## Images and Registries

Docker images are built from Dockerfiles — text files with instructions to assemble an image layer by layer. Images are stored in registries like Docker Hub or GitHub Container Registry. The `docker build` and `docker pull` commands create and download images respectively.

## Common Commands

- `docker run` — create and start a container
- `docker build` — build an image from a Dockerfile
- `docker compose up` — start services defined in docker-compose.yml
- `docker ps` — list running containers
- `docker logs` — view container output
