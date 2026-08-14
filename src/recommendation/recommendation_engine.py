"""
Business recommendation engine.
"""

from __future__ import annotations

from statistics import mean

from src.retrieval.campaign_retriever import CampaignRetriever


class RecommendationEngine:
    """
    Generates business recommendations from similar campaigns.
    """

    def __init__(self):

        self.retriever = CampaignRetriever()

    def recommend(
        self,
        description: str,
        top_k: int = 20,
    ) -> dict:

        campaigns = self.retriever.search(
            description,
            top_k=top_k,
        )

        if not campaigns:
            return {
                "recommendations": [
                    "No similar campaigns found."
                ]
            }

        successful = [
            c for c in campaigns
            if c["state"] == "successful"
        ]

        failed = [
            c for c in campaigns
            if c["state"] == "failed"
        ]

        source = successful if successful else campaigns

        success_rate = len(successful) / len(campaigns)

        avg_goal = mean(
            c["goal"]
            for c in source
        )

        avg_pledged = mean(
            c["pledged"]
            for c in source
        )

        avg_backers = mean(
            c["backers"]
            for c in source
        )

        avg_similarity = mean(
            c["similarity"]
            for c in campaigns
        )

        # ------------------------
        # Category statistics
        # ------------------------

        category_counts = {}

        for campaign in source:

            category = campaign["category"]

            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

        top_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # ------------------------
        # Country statistics
        # ------------------------

        country_counts = {}

        for campaign in source:

            country = campaign["country"]

            country_counts[country] = (
                country_counts.get(country, 0) + 1
            )

        top_countries = sorted(
            country_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # ------------------------
        # Recommendations
        # ------------------------

        recommendations = []

        if success_rate >= 0.70:

            recommendations.append(
                "Very promising niche based on similar campaigns."
            )

        elif success_rate >= 0.50:

            recommendations.append(
                "Moderate probability of success."
            )

        else:

            recommendations.append(
                "Most similar campaigns failed. Improve your positioning before launch."
            )

        recommendations.append(
            f"Recommended funding goal: ${avg_goal:,.0f}"
        )

        recommendations.append(
            f"Successful campaigns raised about ${avg_pledged:,.0f}"
        )

        recommendations.append(
            f"Successful campaigns attracted about {avg_backers:.0f} backers."
        )

        if top_categories:

            recommendations.append(
                f"Best performing category: {top_categories[0][0]}"
            )

        if top_countries:

            recommendations.append(
                f"Most successful country: {top_countries[0][0]}"
            )

        if avg_similarity > 0.75:

            recommendations.append(
                "Your idea is highly similar to existing campaigns."
            )

        elif avg_similarity > 0.60:

            recommendations.append(
                "Your project shares characteristics with previous campaigns."
            )

        else:

            recommendations.append(
                "This appears to be a relatively unique campaign idea."
            )

        return {

            "success_rate": round(success_rate, 2),

            "recommended_goal": round(avg_goal, 2),

            "average_pledged": round(avg_pledged, 2),

            "average_backers": round(avg_backers),

            "average_similarity": round(avg_similarity, 3),

            "top_categories": top_categories[:5],

            "top_countries": top_countries[:5],

            "successful_examples": successful[:5],

            "failed_examples": failed[:5],

            "recommendations": recommendations,
        }