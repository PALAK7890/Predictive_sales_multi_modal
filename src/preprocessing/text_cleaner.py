import re
import unicodedata

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextCleaner:
    """
    Generic text preprocessing transformer.

    Compatible with any pandas Series.
    """

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    # =====================================================
    # Scikit-Learn Compatibility
    # =====================================================

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # =====================================================
    # Public Transform
    # =====================================================

    def transform(self, text_series):

        if not isinstance(text_series, pd.Series):
            raise TypeError(
                "Input must be a pandas Series."
            )

        return text_series.fillna("").apply(self.clean_text)

    # =====================================================
    # Cleaning Pipeline
    # =====================================================

    def clean_text(self, text):

        text = str(text)

        # Unicode normalization
        text = unicodedata.normalize("NFKD", text)

        # lowercase
        text = text.lower()

        # HTML
        text = re.sub(r"<.*?>", " ", text)

        # URLs
        text = re.sub(r"http\\S+|www\\S+", " ", text)

        # emails
        text = re.sub(
            r"\S+@\S+",
            " ",
            text
        )

        # numbers
        text = re.sub(r"\d+", " ", text)

        # punctuation
        text = re.sub(
            r"[^a-z\s]",
            " ",
            text
        )

        # extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        tokens = []

        for word in text.split():

            if word in self.stop_words:
                continue

            if len(word) <= 1:
                continue

            lemma = self.lemmatizer.lemmatize(word)

            tokens.append(lemma)

        return " ".join(tokens)