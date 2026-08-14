"""
Campaign Intelligence Engine.

Combines retrieval with business insights.
"""

from statistics import mean

from src.retrieval.campaign_retriever import CampaignRetriever


class CampaignIntelligence:

    def __init__(self):

        self.retriever = CampaignRetriever()

    def analyze(
        self,
        query: str,
        top_k: int = 100,
    ):

        campaigns = self.retriever.search(
            query,
            top_k=top_k,
        )

        successful = [
            c for c in campaigns
            if c["state"] == "successful"
        ]

        failed = [
            c for c in campaigns
            if c["state"] == "failed"
        ]

        success_rate = len(successful) / len(campaigns)

        avg_goal = (
            mean(c["goal"] for c in successful)
            if successful else 0
        )

        avg_pledged = (
            mean(c["pledged"] for c in successful)
            if successful else 0
        )

        top_categories = {}

        for c in successful:

            cat = c["main_category"]

            top_categories[cat] = (
                top_categories.get(cat, 0) + 1
            )

        top_categories = sorted(
            top_categories.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {

            "retrieved": len(campaigns),

            "successful": len(successful),

            "failed": len(failed),

            "success_rate": round(success_rate * 100, 2),

            "recommended_goal": round(avg_goal, 2),

            "average_pledged": round(avg_pledged, 2),

            "best_categories": top_categories[:5],

            "examples": successful[:5],
        }