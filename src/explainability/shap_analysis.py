"""
Generate SHAP values for the trained XGBoost model.
"""

from pathlib import Path

import joblib
import shap

from scipy.sparse import load_npz


class SHAPAnalyzer:

    def __init__(self):

        self.model = joblib.load(
            "artifacts/models/xgboost.pkl"
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    def explain(
        self,
        sample,
    ):

        if hasattr(sample, "toarray"):
            sample = sample.toarray()

        explanation = self.explainer(sample)

        return explanation.values

    def summary_plot(
        self,
        X,
    ):

        shap.summary_plot(
            self.explainer.shap_values(X),
            X,
            show=False,
        )

    def save(self):

        Path(
            "artifacts/explainability"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.explainer,
            "artifacts/explainability/shap_explainer.pkl",)