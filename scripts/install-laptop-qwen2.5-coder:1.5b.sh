
original="qwen2.5-coder:1.5b-instruct-q5_K_M"
alias_name="qwen2.5-coder:1.5b-q5"
# 1.1GB
# 32K

# completion
# tools

ollama pull "$original"
ollama cp "$original" "$alias_name"
ollama rm "$original"

echo "Ollama: $alias_name listo ($original)."
