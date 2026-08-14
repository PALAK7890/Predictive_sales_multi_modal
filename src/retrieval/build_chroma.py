"""
Build ChromaDB campaign database.
"""

import pandas as pd

from src.embeddings.sentence_transformer import SentenceEmbeddingModel
from src.retrieval.chroma_store import CampaignVectorStore


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        "data/interim/cleaned_text_dataset.csv"
    )

    print("Loading embeddings...")

    embeddings = SentenceEmbeddingModel.load(
        "data/embeddings/campaign_embeddings.pkl"
    )

    print("Creating ChromaDB collection...")

    store = CampaignVectorStore()

    if store.count() > 0:
        print("Existing collection found.")
        print("Resetting collection...")
        store.reset()

    ids = df["id"].astype(str).tolist()

    documents = df["clean_text"].fillna("").tolist()

    metadata = []

    for _, row in df.iterrows():

        metadata.append(
            {
                "title": str(row["name"]),
                "category": str(row["category"]),
                "main_category": str(row["main_category"]),
                "goal": float(row["usd_goal_real"]),
                "pledged": float(row["usd_pledged_real"]),
                "country": str(row["country"]),
                "state": str(row["state"]),
            }
        )

    print("Uploading vectors...")

    store.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadata,
    )

    print()

    print("=" * 50)
    print("ChromaDB Build Complete")
    print("=" * 50)
    print(f"Campaigns Indexed : {store.count():,}")


if __name__ == "__main__":
    main()