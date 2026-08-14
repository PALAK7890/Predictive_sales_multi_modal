"""
Campaign Risk Analyzer
"""

from __future__ import annotations

from src.recommendation.recommendation_engine import RecommendationEngine
from src.recommendation.funding_optimizer import FundingGoalOptimizer


class RiskAnalyzer:
    """
    Analyzes campaign risk using historical similar campaigns.
    """

    def __init__(self):

        self.recommender = RecommendationEngine()
        self.optimizer = FundingGoalOptimizer()

    def analyze(
        self,
        description: str,
    ) -> dict:

        recommendation = self.recommender.recommend(
            description,
            top_k=50,
        )

        funding = self.optimizer.optimize(
            description,
            top_k=50,
        )

        success_rate = recommendation["success_rate"]

        risk_score = 0
        reasons = []
        suggestions = []

        # ------------------------
        # Success Rate
        # ------------------------

        if success_rate < 0.25:

            risk_score += 50

            reasons.append(
                "Very few similar campaigns succeeded."
            )

            suggestions.append(
                "Differentiate your product and strengthen your launch strategy."
            )

        elif success_rate < 0.50:

            risk_score += 25

            reasons.append(
                "Only a moderate number of similar campaigns succeeded."
            )

        # ------------------------
        # Funding Goal
        # ------------------------

        if funding["successful_campaigns"] < 5:

            risk_score += 25

            reasons.append(
                "Very few successful reference campaigns were found."
            )

        elif funding["successful_campaigns"] < 10:

            risk_score += 10

            reasons.append(
                "Limited successful campaigns exist in this niche."
            )

        # ------------------------
        # Category Diversity
        # ------------------------

        if len(recommendation["top_categories"]) <= 2:

            risk_score += 10

            reasons.append(
                "Campaigns are concentrated in only a few categories."
            )

        # ------------------------
        # Risk Label
        # ------------------------

        if risk_score >= 70:

            label = "HIGH"

        elif risk_score >= 40:

            label = "MEDIUM"

        else:

            label = "LOW"

        return {

            "risk_level": label,

            "risk_score": risk_score,

            "success_rate": success_rate,

            "recommended_goal":
                funding["recommended_goal"],

            "average_goal":
                funding["average_goal"],

            "successful_campaigns":
                funding["successful_campaigns"],

            "reasons": reasons,

            "suggestions": suggestions,
        }