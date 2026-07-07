import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

def get_embedding_model() -> str:
    try:
        import streamlit as st
        return st.session_state.get("embed_model") or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    except Exception:
        return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

COLLECTION_NAME = "autoschool_docs"
DB_PATH = "data/chroma"


def get_collection(openai_api_key: str):
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=get_embedding_model(),
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def index_chunks(chunks: list[str], openai_api_key: str):
    collection = get_collection(openai_api_key)
    if collection.count() > 0:
        return  # Already indexed

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        collection.add(documents=batch, ids=ids)


def search(query: str, openai_api_key: str, n_results: int = 5) -> list[str]:
    collection = get_collection(openai_api_key)
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def is_indexed(openai_api_key: str) -> bool:
    try:
        return get_collection(openai_api_key).count() > 0
    except Exception:
        return False


def reset_index():
    import shutil
    shutil.rmtree(DB_PATH, ignore_errors=True)
