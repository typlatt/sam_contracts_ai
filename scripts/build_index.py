import os
import faiss
import numpy as np
import openai
from typing import List
from constants import INDEX_PATH, DESC_PATH
from utills import (
    load_contracts,
    batch_texts_by_token_limit,
    embed_openai,
    truncate_text_to_token_limit,
)

openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_PATH = INDEX_PATH
DESC_PATH = DESC_PATH

if __name__ == "__main__":
    print("Loading contracts...")
    df = load_contracts()
    descriptions = df["Description"].tolist()

    print("Batching descriptions for embedding...")
    batches = batch_texts_by_token_limit(descriptions)
    embeddings = []
    for i, batch in enumerate(batches):
        print(f"Embedding batch {i+1}/{len(batches)} (size: {len(batch)})...")
        embeddings.extend(embed_openai(batch))

    print("Building FAISS index...")
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32")) 

    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving descriptions to {DESC_PATH}...")
    df[["Description"]].to_csv(DESC_PATH, index=False)

    print("Done! You can now run queries using the saved index and descriptions.") 