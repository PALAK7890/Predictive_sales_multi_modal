"""
Predict Kickstarter campaign success using XGBoost.
"""

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.embeddings.sentence_transformer import SentenceEmbeddingModel
from src.preprocessing.xgb_preprocessor import XGBPreprocessor


class CampaignPredictor:

    def __init__(self):
        self.embedder = SentenceEmbeddingModel()
        self.preprocessor = XGBPreprocessor().load()
        self.model = joblib.load("artifacts/models/xgboost.pkl")

    def get_features(
        self,
        title,
        category,
        main_category,
        country,
        currency,
        goal,
        duration=30,
    ):
        """
        Generates and fuses embedding and tabular features for a campaign.
        """
        today = pd.Timestamp.today()
        df = pd.DataFrame(
            {
                "name": [title],
                "goal": [goal],
                "usd_goal_real": [goal],
                "country": [country],
                "currency": [currency],
                "category": [category],
                "main_category": [main_category],
                "backers": [0],
                "launched": [today],
                "deadline": [
                    today + pd.Timedelta(days=duration)
                ],
                "clean_text": [title],
            }
        )

        embedding = self.embedder.embed_query(title)
        X_embed = csr_matrix(embedding.reshape(1, -1))
        X_tabular = self.preprocessor.transform(df)
        X = hstack([X_embed, X_tabular])
        return X

    def predict(
        self,
        title,
        category,
        main_category,
        country,
        currency,
        goal,
        duration=30,
    ):
        """
        Predicts the campaign's success probability.
        """
        X = self.get_features(
            title=title,
            category=category,
            main_category=main_category,
            country=country,
            currency=currency,
            goal=goal,
            duration=duration,
        )

        probability = self.model.predict_proba(X)[0][1]
        prediction = "Successful" if probability >= 0.5 else "Failed"

        return {
            "prediction": prediction,
            "success_probability": round(probability * 100, 2),
            "failure_probability": round((1 - probability) * 100, 2),
        }