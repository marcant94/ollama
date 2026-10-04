# Ollama — Guía del proyecto

Servicio Ollama dockerizado (CPU o GPU NVIDIA/AMD) que da servicio a chat-hub (repositorio hermano `../chathub`) y a otros clientes locales.

## Arquitectura

- Un único servicio `ollama` con volumen persistente (`ollama_data`) para los modelos descargados.
- El compose publica el puerto 11434 y se une a la red `proxylan`: el `docker-compose.yml` del repo `chathub` reutiliza esa red y accede vía `http://ollama:11434`.

## Archivos clave

| Archivo | Responsabilidad |
|---------|-----------------|
| `docker-compose.yml` | Servicio base de Ollama (red `proxylan`, volumen de modelos) |
| `docker-compose.nvidia.yml` / `docker-compose.amd.yml` | Overrides para GPU NVIDIA / AMD-Intel |
| `Dockerfile.ollama.cpu` / `Dockerfile.ollama.gpu` | Imagen según hardware; `OLLAMA_DOCKERFILE` elige |
| `dkr-compilar-ollama.sh` | Carga el `.env` y hace `up -d --build` según `OLLAMA_GPU` (nvidia/amd/vacío) |
| `dkr-ver-log-ollama.sh` | Logs del contenedor |
| `dkr-acceder-ollama.sh` | Shell dentro del contenedor |
| `scripts/install-*.sh` | Descarga de modelos por máquina (torre, laptop, vps, orquestador) |
| `tools/bench-ollama.py` | Benchmark de tok/s de los modelos instalados (unload, warmup, N mediciones, resumen). Guarda el historial en `tools/bench-historial.json` (ignorado en git, específico de la máquina) |
| `tools/gestion-ollama.py` | Menú interactivo: listar instalados/cargados, arrancar con ctx defecto/4K/8K/16K, descargar y borrar modelos |
| `tools/estimar-modelo.py` | Estima tok/s de un modelo sin descargarlo (URL de ollama.com o HuggingFace, o `params/tamaño`), calibrado con el historial local |
| `.env-base/.env-vps` | Plantilla de variables (límites de RAM/CPU, GPU) |

## Variables de entorno (.env, no se commitea)

- `OLLAMA_DOCKERFILE`: Dockerfile a usar (cpu/gpu).
- `OLLAMA_GPU`: `nvidia` | `amd` | vacío (CPU) — lo usa `dkr-compilar-ollama.sh`.
- `OLLAMA_NUM_PARALLEL`: peticiones concurrentes.
- `OLLAMA_MEM_LIMIT` / `OLLAMA_CPUS`: límites de recursos (útil en VPS).

## Despliegue

- Compilar/arrancar: `sh dkr-compilar-ollama.sh` (elige compose override por GPU).
- Ver logs: `sh dkr-ver-log-ollama.sh`.
- Instalar un modelo nuevo: copiar el patrón de `scripts/install-torre-*.sh` (`ollama pull` + alias corto con `ollama create` si conviene).
- Benchmark: `python3 tools/bench-ollama.py` (desde el host, con Ollama levantado).
- Gestionar modelos: `python3 tools/gestion-ollama.py` (menú: listar instalados/cargados, arrancar con ctx por defecto/4K/8K/16K, descargar de memoria, borrar).
- Estimar antes de descargar: `python3 tools/estimar-modelo.py` (sin argumentos abre un bucle interactivo donde pegas URLs de ollama.com/HuggingFace o datos `4B 2.5GB`; con argumentos hace una única estimación).
- El compose del repo `chathub` necesita que este servicio esté levantado para resolver `http://ollama:11434` (misma red `proxylan`).

## Notas

- El volumen `ollama_data` conserva los modelos entre recreaciones del contenedor.
- Ollama es inconsistente reportando capabilities en `/api/tags` (p. ej. `ling-3.0-tiny` no reporta thinking aunque lo tiene); chat-hub lo compensa consultando `POST /api/show`.
