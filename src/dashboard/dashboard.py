from src.retrieval.campaign_retriever import CampaignRetriever


class SimilarCampaignDashboard:
    """
    Backend for the Similar Campaign Dashboard.
    """

    def __init__(self):

        self.retriever = CampaignRetriever()

    def search(
        self,
        query: str,
        top_k: int = 10,
    ):

        campaigns = self.retriever.search(
            query=query,
            top_k=top_k,
        )

        total = len(campaigns)

        successful = sum(
            1
            for c in campaigns
            if c["state"] == "successful"
        )

        failed = total - successful

        success_rate = round(
            successful / total * 100,
            2,
        ) if total else 0

        return {
            "summary": {
                "retrieved": total,
                "successful": successful,
                "failed": failed,
                "success_rate": success_rate,
            },
            "campaigns": campaigns,
        }