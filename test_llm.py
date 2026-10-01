import ollama

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Explain what an AI agent is in one sentence."
        }
    ]
)

print("\nQwen3 response:")
print(response["message"]["content"])