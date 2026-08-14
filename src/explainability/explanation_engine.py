"""
Human-readable SHAP explanations.
"""

import joblib
import numpy as np
from scipy.sparse import load_npz

from src.explainability.shap_analysis import SHAPAnalyzer


class ExplanationEngine:

    def __init__(self):
        self.analyzer = SHAPAnalyzer()
        try:
            self.preprocessor = joblib.load("artifacts/preprocessors/xgb_preprocessor.pkl")
            self.tabular_feature_names = self.preprocessor.get_feature_names_out()
        except Exception as e:
            print(f"Warning: Could not load preprocessor or feature names: {e}")
            self.tabular_feature_names = None

    def _get_clean_feature_name(self, index: int) -> str:
        """
        Maps a feature index to a clean human-readable name.
        """
        if index < 384:
            return f"Campaign Description Semantics (Dim {index})"

        if self.tabular_feature_names is None:
            return f"Feature {index}"

        tab_idx = index - 384
        if tab_idx >= len(self.tabular_feature_names):
            return f"Feature {index}"

        raw_name = self.tabular_feature_names[tab_idx]

        # Clean up scikit-learn prefix
        if raw_name.startswith("numeric__"):
            clean = raw_name[len("numeric__"):]
            mapping = {
                "goal": "Campaign Goal (Local Currency)",
                "goal_log": "Campaign Goal (Log scale)",
                "usd_goal_real": "Campaign Goal (USD)",
                "usd_goal_real_log": "Campaign Goal (USD, Log scale)",
                "campaign_duration": "Campaign Duration (Days)",
                "launch_year": "Launch Year",
                "launch_month": "Launch Month",
                "launch_day": "Launch Day of Month",
                "launch_weekday": "Launch Day of Week",
                "launch_quarter": "Launch Quarter",
                "title_length": "Title Length (Characters)",
                "title_word_count": "Title Word Count",
                "goal_per_day": "Funding Goal Per Day (USD)",
                "backers": "Number of Backers"
            }
            return mapping.get(clean, clean.replace("_", " ").title())

        elif raw_name.startswith("categorical__"):
            clean = raw_name[len("categorical__"):]
            if clean.startswith("country_"):
                val = clean[len("country_"):]
                if val == 'N,0"':
                    val = "Unknown"
                return f"Country: {val}"
            elif clean.startswith("currency_"):
                val = clean[len("currency_"):]
                return f"Currency: {val}"
            elif clean.startswith("category_"):
                val = clean[len("category_"):]
                return f"Category: {val}"
            elif clean.startswith("main_category_"):
                val = clean[len("main_category_"):]
                return f"Main Category: {val}"
            return clean.replace("_", " ").title()

        return raw_name

    def explain_instance(self, X_sample, top_k: int = 10) -> dict:
        """
        Explains a specific fused feature row, returning clean positive and negative factors.
        """
        # Convert sparse matrix to dense array
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()

        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = self.analyzer.explain(X_sample)
        shap_values = np.asarray(shap_values).flatten()

        indices = np.arange(len(shap_values))

        # Filter and sort positive factors (SHAP > 1e-5)
        pos_mask = shap_values > 1e-5
        pos_indices = indices[pos_mask]
        pos_values = shap_values[pos_mask]
        pos_sorted_idx = np.argsort(pos_values)[::-1]  # largest positive first

        positive_factors = []
        for idx in pos_sorted_idx[:top_k]:
            feature_idx = int(pos_indices[idx])
            positive_factors.append({
                "feature": self._get_clean_feature_name(feature_idx),
                "shap_value": float(pos_values[idx]),
                "index": feature_idx
            })

        # Filter and sort negative factors (SHAP < -1e-5)
        neg_mask = shap_values < -1e-5
        neg_indices = indices[neg_mask]
        neg_values = shap_values[neg_mask]
        neg_sorted_idx = np.argsort(neg_values)  # most negative first

        negative_factors = []
        for idx in neg_sorted_idx[:top_k]:
            feature_idx = int(neg_indices[idx])
            negative_factors.append({
                "feature": self._get_clean_feature_name(feature_idx),
                "shap_value": float(neg_values[idx]),
                "index": feature_idx
            })

        return {
            "positive_factors": positive_factors,
            "negative_factors": negative_factors
        }

    def explain_prediction(self, index: int, top_k: int = 10) -> dict:
        """
        Explains a campaign from the training dataset.
        """
        X = load_npz("artifacts/fusion/X_embedding_fused.npz")
        sample = X[index]
        explanation = self.explain_instance(sample, top_k=top_k)
        explanation["sample"] = index
        return explanation