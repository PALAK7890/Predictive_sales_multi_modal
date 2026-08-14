"""
Analyze historical success statistics from retrieved campaigns.
"""

from statistics import mean, median


class SuccessAnalyzer:
    """
    Computes business statistics from similar campaigns.
    """

    def analyze(self, campaigns: list[dict]) -> dict:

        successful = [
            c for c in campaigns
            if c["state"] == "successful"
        ]

        failed = [
            c for c in campaigns
            if c["state"] != "successful"
        ]

        total = len(campaigns)

        success_rate = (
            len(successful) / total
            if total else 0
        )

        def avg(values):
            return round(mean(values), 2) if values else 0

        def med(values):
            return round(median(values), 2) if values else 0

        success_goals = [
            c["goal"] for c in successful
        ]

        success_pledged = [
            c["pledged"] for c in successful
        ]

        success_backers = [
            c["backers"] for c in successful
        ]

        failure_goals = [
            c["goal"] for c in failed
        ]

        return {
            "total_campaigns": total,

            "successful_campaigns": len(successful),

            "failed_campaigns": len(failed),

            "success_rate": round(success_rate, 3),

            "average_success_goal": avg(success_goals),

            "median_success_goal": med(success_goals),

            "average_success_pledged": avg(success_pledged),

            "median_success_pledged": med(success_pledged),

            "average_success_backers": avg(success_backers),

            "average_failure_goal": avg(failure_goals),

            "successful_examples": successful[:5],

            "failed_examples": failed[:5],
        }