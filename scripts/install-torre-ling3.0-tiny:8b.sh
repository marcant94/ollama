
original="hf.co/inclusionAI/Ling-3.0-tiny-GGUF:Q4_K_M"
alias_name="ling3.0-tiny:8b"
# 4.8GB
# 128K

# completion
# tools
# thinking

ollama pull "$original"
ollama cp "$original" "$alias_name"
ollama rm "$original"

echo "Ollama: $alias_name listo ($original)."
