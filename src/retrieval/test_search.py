from src.retrieval.campaign_retriever import CampaignRetriever

retriever = CampaignRetriever()

results = retriever.search(
    "AI powered fitness app",
    top_k=5,
)

print("\nRetrieved Campaigns\n")

for i, campaign in enumerate(results, start=1):
    print(f"{i}. {campaign['name']}")
    print(f"Category : {campaign['category']}")
    print(f"State    : {campaign['state']}")
    print(f"Score    : {campaign['score']:.4f}")
    print("-" * 60)