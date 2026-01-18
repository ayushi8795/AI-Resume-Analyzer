from __future__ import annotations
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def build_chroma(
    texts: List[str],
    metadatas: List[Dict],
    persist_dir: str,
    collection_name: str,
    embed_model: str,
) -> Chroma:
    embeddings = OpenAIEmbeddings(model=embed_model)
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    vs.persist()
    return vs

def load_chroma(
    persist_dir: str,
    collection_name: str,
    embed_model: str,
) -> Chroma:
    embeddings = OpenAIEmbeddings(model=embed_model)
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
