"""
Generate dense embeddings for Kickstarter campaigns.
"""

import pandas as pd

from src.embeddings.sentence_transformer import SentenceEmbeddingModel


def main():

    df = pd.read_csv(
        "data/interim/cleaned_text_dataset.csv"
    )

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    model = SentenceEmbeddingModel()

    embeddings = model.encode(texts)

    model.save(
        embeddings,
        "data/embeddings/campaign_embeddings.pkl",
    )

    print("Embeddings Shape:", embeddings.shape)


if __name__ == "__main__":
    main()