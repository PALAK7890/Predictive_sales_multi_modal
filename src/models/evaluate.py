from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


class ModelEvaluator:

    def __init__(self):
        self.figures_dir = Path("reports/figures")
        self.tables_dir = Path("reports/tables")

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob),
        }

        print("\nEvaluation Results")
        print("-" * 40)

        for metric, value in metrics.items():
            print(f"{metric:<12}: {value:.4f}")

        print("\nClassification Report\n")
        print(classification_report(y_test, y_pred))

        pd.DataFrame([metrics]).to_csv(
            self.tables_dir / "metrics.csv",
            index=False,
        )

        with open(self.tables_dir / "classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred))

        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc(model, X_test, y_test)
        self.plot_pr(model, X_test, y_test)
        self.feature_importance(model)

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            cmap="Blues",
            ax=ax,
        )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "confusion_matrix.png")
        plt.close()

    def plot_roc(self, model, X_test, y_test):
        fig, ax = plt.subplots(figsize=(6, 5))

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax,
        )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "roc_curve.png")
        plt.close()

    def plot_pr(self, model, X_test, y_test):
        fig, ax = plt.subplots(figsize=(6, 5))

        PrecisionRecallDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax,
        )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "precision_recall_curve.png")
        plt.close()

    def feature_importance(self, model):

        print("\nAnalyzing feature importance...")

        tfidf = joblib.load("models/vectorizers/tfidf.pkl")
        tabular = joblib.load("models/encoders/tabular_preprocessor.pkl")

        text_features = list(tfidf.get_feature_names_out())

        raw_tabular_features = tabular.get_feature_names_out()

        tabular_features = [
            feature.replace("numeric__", "").replace("categorical__", "")
            for feature in raw_tabular_features
        ]

        feature_types = []

        for feature in raw_tabular_features:
            if feature.startswith("numeric__"):
                feature_types.append("Numeric")
            else:
                feature_types.append("Category")

        feature_names = text_features + tabular_features

        feature_types = ["Text"] * len(text_features) + feature_types

        importance = pd.DataFrame({
            "Feature": feature_names,
            "Type": feature_types,
            "Coefficient": model.coef_[0]
        })

        importance["Importance"] = importance["Coefficient"].abs()

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        importance.to_csv(
            self.tables_dir / "feature_importance.csv",
            index=False,
        )

        print("\nTop 15 Features\n")
        print(
            importance[
                ["Feature", "Type", "Coefficient"]
            ].head(15)
        )


    
    def plot_feature_importance(self, importance):

        plt.figure(figsize=(10,7))

        data = importance.sort_values("Coefficient")

        plt.barh(
            data["Feature"],
            data["Coefficient"]
        )

        plt.xlabel("Coefficient")
        plt.title("Top Feature Importance")

        plt.tight_layout()

        plt.savefig(
            self.figures_dir / "feature_importance.png"
        )

        plt.close()