"""
Funding Goal Optimizer
"""

from __future__ import annotations

from statistics import median

from src.retrieval.campaign_retriever import CampaignRetriever


class FundingGoalOptimizer:
    """
    Suggests an optimal funding goal based on
    successful similar Kickstarter campaigns.
    """

    def __init__(self):

        self.retriever = CampaignRetriever()

    def optimize(
        self,
        description: str,
        top_k: int = 50,
    ) -> dict:

        campaigns = self.retriever.search(
            description,
            top_k=top_k,
        )

        successful = [
            c
            for c in campaigns
            if c["state"] == "successful"
        ]

        if len(successful) == 0:

            return {
                "recommended_goal": None,
                "minimum_goal": None,
                "maximum_goal": None,
                "average_goal": None,
                "reason":
                "No successful similar campaigns were found."
            }

        goals = sorted(
            c["goal"]
            for c in successful
        )

        average_goal = sum(goals) / len(goals)

        return {

            "recommended_goal": round(
                median(goals),
                2,
            ),

            "minimum_goal": round(
                goals[0],
                2,
            ),

            "maximum_goal": round(
                goals[-1],
                2,
            ),

            "average_goal": round(
                average_goal,
                2,
            ),

            "successful_campaigns": len(successful),

            "total_compared": len(campaigns),
        }