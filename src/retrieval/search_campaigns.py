"""
Semantic campaign search using ChromaDB.
"""

from src.embeddings.sentence_transformer import SentenceEmbeddingModel
from src.retrieval.chroma_store import CampaignVectorStore


def search(query: str, k: int = 5):

    model = SentenceEmbeddingModel()

    embedding = model.encode([query])[0]

    store = CampaignVectorStore()

    results = store.search(
        embedding,
        n_results=k,
    )

    for i in range(k):

        meta = results["metadatas"][0][i]

        score = 1 - results["distances"][0][i]

        print("-" * 60)

        print(f"Rank        : {i+1}")
        print(f"Similarity  : {score:.3f}")
        print(f"Title       : {meta['title']}")
        print(f"Category    : {meta['main_category']}")
        print(f"Goal        : ${meta['goal']:,.0f}")
        print(f"Raised      : ${meta['pledged']:,.0f}")
        print(f"Outcome     : {meta['state']}")