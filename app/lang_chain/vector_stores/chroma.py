import os

import chromadb
from chromadb.api import ClientAPI
from abc import ABC, abstractmethod
from pathlib import Path
from langchain_community.vectorstores import Chroma as VS_Chroma


class Chroma(ABC):
    def __init__(self):
        self.storage_path = Path(__file__).parents[3] / 'storage' / 'chroma'
        self.client = chromadb.PersistentClient(path=self.storage_path.__str__())

    def get_client(self) -> ClientAPI:
        return self.client

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        collection = self.get_collection(collection_name)
        if collection:
            return collection
        return self.client.create_collection(name=collection_name)

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(name=collection_name)

    def get_collection_list(self):
        return self.client.list_collections()

    @abstractmethod
    def get_embedding(self):
        pass

    @abstractmethod
    def get_cached_embedding(self):
        pass

    def get_db(self, collection_name: str):
        return VS_Chroma(
            client=self.client,
            persist_directory=collection_name,
            collection_name=collection_name,
            embedding_function=self.get_cached_embedding()
        )
