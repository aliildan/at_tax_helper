import os

import chromadb
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from pathlib import Path
from app.lang_chain.vector_stores.chroma import Chroma
from langchain_community.vectorstores import Chroma as VS_Chroma


class OllamaEmbedding(Chroma):
    def __init__(self, model: str = "llama3"):
        self.storage_path = Path(__file__).parents[3] / 'storage' / 'embeddings_cache'
        self.embedding = OllamaEmbeddings(model=model, show_progress=True)
        self.store = LocalFileStore(self.storage_path.__str__())
        self.cached_embedding = CacheBackedEmbeddings.from_bytes_store(self.embedding, self.store,
                                                                       namespace=self.embedding.model)
        super().__init__()

    def get_embedding(self) -> OllamaEmbeddings:
        return self.embedding

    def get_cached_embedding(self):
        return self.cached_embedding

    def save_documents(self, collection_name: str, documents):
        self.cached_embedding.embed_documents(self.get_cached_embedding_elements(documents))
        VS_Chroma.from_documents(
            client=self.client,
            documents=documents,
            embedding=self.cached_embedding,
            collection_name=collection_name,
            persist_directory=collection_name
        )

    def save_texts(self, collection_name: str, texts):
        self.cached_embedding.embed_documents(self.get_cached_embedding_elements(documents=texts))
        VS_Chroma.from_texts(
            client=self.client,
            texts=texts,
            embedding=self.cached_embedding,
            collection_name=collection_name,
            persist_directory=collection_name
        )

    def get_query_result(self, collection_name: str, query: str):
        return self.client.get_collection(name=collection_name).query(
            query_embeddings=self.cached_embedding.embed_query(query))

    def get_db(self, collection_name: str):
        return super().get_db(collection_name)

    def get_cached_embedding_elements(self, documents) -> list:
        docs = []
        for document in documents:
            docs.append(document.page_content)
        return docs
