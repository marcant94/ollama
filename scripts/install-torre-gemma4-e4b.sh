
original="hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf"
alias_name="gemma4-e4b:7b"
# 5.15GB
# 128K

# completion
# tools
# vision
# audio

ollama pull "$original"
ollama cp "$original" "$alias_name"
ollama rm "$original"

echo "Ollama: $alias_name listo ($original)."
