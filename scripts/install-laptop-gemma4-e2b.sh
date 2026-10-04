
original="hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf"
alias_name="gemma4-e2b:5b"
# 4.3GB
# 128K

# completion
# tools
# vision
# audio

ollama pull "$original"
ollama cp "$original" "$alias_name"
ollama rm "$original"

echo "Ollama: $alias_name listo ($original)."
