
# original="maternion/ling-3.0-tiny:8b-Q4_K_M"
original="hf.co/inclusionAI/Ling-3.0-tiny-GGUF:Q4_K_M"
alias_name="ling3.0-tiny:8b"
# 4.8GB
# 128K

# completion
# tools
# thinking (Ollama no lo reporta en /api/tags para este modelo; chat-hub
# consulta POST /api/show por modelo y une las capabilities)

ollama pull "$original"
ollama cp "$original" "$alias_name"
ollama rm "$original"

echo "Ollama: $alias_name listo ($original)."
