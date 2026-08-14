from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from configs.config import TFIDF_CONFIG


class TFIDFVectorizer:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(**TFIDF_CONFIG)

        self.model_dir = Path("models/vectorizers")
        self.report_dir = Path("reports/tables")

    def fit(self, text):

        self.vectorizer.fit(text)
        return self

    def transform(self, text):

        return self.vectorizer.transform(text)

    def fit_transform(self, text):

        return self.vectorizer.fit_transform(text)

    def save(self):

        self.model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.vectorizer,
            self.model_dir / "tfidf.pkl",
        )

    def save_vocabulary(self):

        self.report_dir.mkdir(parents=True, exist_ok=True)

        vocabulary = pd.DataFrame(
            {
                "feature": self.vectorizer.get_feature_names_out()
            }
        )

        vocabulary.to_csv(
            self.report_dir / "vocabulary.csv",
            index=False,
        )

    def vocabulary_size(self):

        return len(self.vectorizer.get_feature_names_out())