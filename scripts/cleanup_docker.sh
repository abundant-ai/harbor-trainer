#!/bin/bash
# Clean up Docker resources from failed/stuck trials

echo "Stopping all running containers..."
docker ps -q | xargs -r docker stop

echo "Removing all containers..."
docker ps -aq | xargs -r docker rm

echo "Pruning Docker system..."
docker system prune -f

echo "Pruning Docker volumes..."
docker volume prune -f

echo "Pruning Docker networks..."
docker network prune -f

echo "Docker cleanup complete!"
echo "Current container count: $(docker ps -a | wc -l)"

