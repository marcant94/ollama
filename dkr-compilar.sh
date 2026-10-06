#!/bin/sh

# Carga el .env (docker compose NO exporta esas variables al shell del script)
if [ -f "$(dirname "$0")/.env" ]; then
    set -a
    . "$(dirname "$0")/.env"
    set +a
fi

# OLLAMA_GPU controla el override de GPU:
#   nvidia -> docker-compose.nvidia.yml (GPU NVIDIA)
#   amd    -> docker-compose.amd.yml    (GPU AMD/Intel)
#   vacío  -> solo CPU (sin override)
case "${OLLAMA_GPU:-}" in
    nvidia)
        echo "Modo NVIDIA GPU"
        docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d --build ollama
        ;;
    amd)
        echo "Modo AMD/Intel GPU"
        docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d --build ollama
        ;;
    *)
        echo "Modo CPU"
        docker compose up -d --build ollama
        ;;
esac

docker image prune -f
