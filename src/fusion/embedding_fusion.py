"""
Embedding + Tabular Feature Fusion
"""

from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import csr_matrix, hstack, save_npz

from src.preprocessing.xgb_preprocessor import XGBPreprocessor


def main():

    print("=" * 60)
    print("Embedding Feature Fusion")
    print("=" * 60)

    print("\nLoading embeddings...")

    embeddings = joblib.load("data/embeddings/campaign_embeddings.pkl")

    print("Embedding Shape:", embeddings.shape)

    print("\nLoading dataset...")

    df = pd.read_csv("data/interim/cleaned_text_dataset.csv")

    print("Dataset Shape:", df.shape)

    print("\nLoading XGBoost preprocessor...")

    processor = XGBPreprocessor().load()

    X_tabular = processor.transform(df)

    print("Tabular Shape:", X_tabular.shape)

    X_embeddings = csr_matrix(embeddings)

    print("\nFusing features...")

    X_fused = hstack([X_embeddings,X_tabular,])

    print("Final Shape:", X_fused.shape)

    Path("artifacts/fusion").mkdir( parents=True,exist_ok=True,)

    save_npz( "artifacts/fusion/X_embedding_fused.npz",X_fused,)

    print("\nFusion Complete!")

    print("Saved:", "artifacts/fusion/X_embedding_fused.npz",)


if __name__ == "__main__":
    main()