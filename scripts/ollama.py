import requests
from typing import List

# === Ask with Ollama Chat ===
def ask_ollama(context: str, question: str, model="llama3.2") -> str:
    # model = 'llama3.2'
    model = 'deepseek-r1:1.5b'
    prompt = f"You are a helpful assistant. Use the following contract summaries to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return res.json().get("response")


def embed_ollama(text: str) -> List[float]:
    response = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return response.json()["embedding"]