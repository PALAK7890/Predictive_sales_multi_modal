"""
Semantic retrieval over Kickstarter campaigns.
"""

from __future__ import annotations

import pandas as pd

from src.embeddings.sentence_transformer import SentenceEmbeddingModel
from src.retrieval.chroma_store import CampaignVectorStore


class CampaignRetriever:
    """
    Retrieves semantically similar Kickstarter campaigns.
    """

    def __init__(
        self,
        collection_name: str = "kickstarter_campaigns",
        data_path: str = "data/interim/cleaned_text_dataset.csv",

    ):

        self.embedder = SentenceEmbeddingModel()
        self.store = CampaignVectorStore()

        self.df = pd.read_csv(data_path)
      

        self.lookup = (
            self.df
            .set_index("id")
)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_embedding = self.embedder.embed_query(query)

        results = self.store.query(
            embedding=query_embedding,
            top_k=top_k,
        )

        print("RESULT KEYS:", results.keys())
        print("IDS:", results["ids"])
        print("DISTANCES:", results["distances"])

        campaigns = []

        ids = results["ids"][0]
        distances = results["distances"][0]

        print("NUMBER OF RESULTS:", len(ids))

        for campaign_id, distance in zip(ids, distances):

            print("PROCESSING ID:", campaign_id)

            campaign_id = int(campaign_id)

            row = self.lookup.loc[campaign_id]

            print("FOUND:", row["name"])

            campaigns.append(
    {
        "id": campaign_id,
        "title": row["name"],
        "category": row["category"],
        "main_category": row["main_category"],
        "goal": float(row["usd_goal_real"]),
        "pledged": float(row["usd_pledged_real"]),
        "backers": int(row["backers"]),
        "country": row["country"],
        "state": row["state"],
        "similarity": round(1 - distance, 4),
    }
)

        print("FINAL:", campaigns)

        return campaigns