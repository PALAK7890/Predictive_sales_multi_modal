"""
ChromaDB vector store for Kickstarter campaigns.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings


class CampaignVectorStore:

    def __init__(
        self,
        persist_directory: str = "artifacts/chromadb",
        collection_name: str = "kickstarter_campaigns",
    ):

        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            },
        )

    def add(
        self,
        ids,
        embeddings,
        documents,
        metadatas,
        batch_size: int = 5000,
    ):
        """
        Add documents to ChromaDB in batches.
        """

        total = len(ids)

        for start in range(0, total, batch_size):

            end = min(start + batch_size, total)

            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

            print(
                f"Indexed {end:,}/{total:,} campaigns",
                end="\r",
            )

        print()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):

        query_embedding = self.embedder.embed_query(query)

        results = self.store.query(
            embedding=query_embedding,
            top_k=top_k,
        )

        print(results)

        return []

    def count(self):

        return self.collection.count()

    def reset(self):

        self.client.delete_collection(
            "kickstarter_campaigns"
        )

        self.collection = self.client.get_or_create_collection(
            "kickstarter_campaigns"
        )
    def query(
        self,
        embedding,
        top_k: int = 5,
    ):

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )