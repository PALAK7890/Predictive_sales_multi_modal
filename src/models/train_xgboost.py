"""
Train XGBoost on Embedding Fusion Features
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from xgboost import XGBClassifier


def main():

    print("=" * 60)
    print("Training XGBoost")
    print("=" * 60)

    X = load_npz(
        "artifacts/fusion/X_embedding_fused.npz"
    )

    df = pd.read_csv(
        "data/interim/cleaned_text_dataset.csv"
    )

    y = (
        df["state"] == "successful"
    ).astype(int)

    print("Dataset:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    print("\nTraining...\n")

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(
        X_test
    )

    print(classification_report(
        y_test,
        pred,
    ))

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            pred,
        )
    )

    Path(
        "artifacts/models"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        "artifacts/models/xgboost.pkl",
    )

    print("\nSaved:")
    print("artifacts/models/xgboost.pkl")


if __name__ == "__main__":
    main()