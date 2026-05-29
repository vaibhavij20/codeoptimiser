import requests

r = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5-coder:3b",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": "hello"
            }
        ]
    },
    timeout=120
)

print(r.status_code)
print(r.json())