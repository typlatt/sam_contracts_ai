# rag_qa.py
import os
import faiss
import numpy as np
import pandas as pd
import duckdb
import openai
import tiktoken
from typing import List


# Make sure this is set in your environment
openai.api_key = os.getenv("OPENAI_API_KEY")


# === Load Data ===
def load_contracts() -> pd.DataFrame:
    conn = duckdb.connect("../data/sam.duckdb")
    df = conn.sql("SELECT * FROM contracts").df()
    return df


#  Switched to batching requests to openai vs. streaming requests
def embed_openai(text: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        model="text-embedding-3-small", input=text)
    return [d.embedding for d in response.data]


# TODO: Implement this to handle stopwords, punctuation, etc. (hitting OPENAI's limit on tokens)
def prepare_openai_embeddings(texts: List[str]):
    pass


# === Truncate Text to Token Limit ===
def truncate_text_to_token_limit(text, token_limit=8192, model="text-embedding-3-small"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= token_limit:
        return text
    print(
        f"[Warning] Truncating text from {len(tokens)} to {token_limit} tokens.")
    truncated = enc.decode(tokens[:token_limit])
    return truncated


# === Embedding Batching Helper ===
def batch_texts_by_token_limit(
    texts,
    max_tokens=32768,
    model="text-embedding-3-small",
    max_single_text_tokens=8192,
    truncate_long_texts=True,
):
    enc = tiktoken.encoding_for_model(model)

    def count_tokens(text):
        return len(enc.encode(text))

    batches = []
    current_batch = []
    current_tokens = 0
    for text in texts:
        tokens = count_tokens(text)
        # TODO: This is a hack to truncate long texts. It should be done in a more elegant way.
        if tokens > max_single_text_tokens:
            if truncate_long_texts:
                text = truncate_text_to_token_limit(
                    text, max_single_text_tokens, model)
                tokens = count_tokens(text)
            else:
                print(
                    f"[Warning] Skipping text with {tokens} tokens (exceeds {max_single_text_tokens} token limit for a single input)"
                )
                continue
        if current_tokens + tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches


# === Build FAISS Index ===
def build_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))  # type: ignore
    return index


# === Context Retrieval ===
def get_context(index: faiss.IndexFlatL2, df: pd.DataFrame, query_embedding: List[float], top_k=5) -> str | None:
    D, I = index.search(np.array([query_embedding]).astype(
        "float32"), top_k)  # type: ignore
    return "\n\n".join(df.iloc[i]["Description"] for i in I[0])


# === Ask with OpenAI Chat ===
def ask_openai(context: str, question: str) -> str | None:
    prompt = f"""You are a helpful assistant. Use the following contract summaries to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    response = openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
    return response["choices"][0]["message"]["content"]
