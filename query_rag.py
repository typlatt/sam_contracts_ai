import os
import faiss
import numpy as np
import pandas as pd
import openai
from scripts.utills import (
    batch_texts_by_token_limit,
    embed_openai,
    truncate_text_to_token_limit,
    ask_openai,
)
from scripts.constants import INDEX_PATH, DESC_PATH

openai.api_key = os.getenv("OPENAI_API_KEY")

INDEX_PATH = INDEX_PATH
DESC_PATH = DESC_PATH

if __name__ == "__main__":
    print("Loading FAISS index and descriptions...")
    index = faiss.read_index(INDEX_PATH)
    df = pd.read_csv(DESC_PATH)

    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Embed the question
        # Truncate if needed
        question_trunc = truncate_text_to_token_limit(question)
        question_embedding = embed_openai([question_trunc])[0]

        # Retrieve context
        D, I = index.search(np.array([question_embedding]).astype("float32"), 5)
        context = "\n\n".join(df.iloc[i]["Description"] for i in I[0])

        # Get answer
        answer = ask_openai(context, question)
        print("\n--- Answer ---\n ", answer)
