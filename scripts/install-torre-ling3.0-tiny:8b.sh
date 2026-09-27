
version="maternion/ling-3.0-tiny:8b-Q4_K_M" # ling-3.0-tiny:8b-UD-Q4_K_XL
alias_name="ling-3.0-tiny:8b"
# 4.8GB
# 128K

# completion
# tools
# thinking (Ollama no lo reporta en /api/tags para este modelo; chat-hub
# consulta POST /api/show por modelo y une las capabilities)

ollama pull "$version"

# Alias corto: apunta a las mismas capas (no duplica la descarga).
printf 'FROM %s\n' "$version" > /tmp/Modelfile.ling
ollama create "$alias_name" -f /tmp/Modelfile.ling
rm -f /tmp/Modelfile.ling

# Borrar el original: ollama rm solo elimina SU manifest; las capas siguen
# referenciadas por el alias. Así /api/tags lista un solo ling.
ollama rm "$version"

echo "Ollama: $alias_name listo (from $version)."

