import re
import unicodedata

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextCleaner:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        self.html_pattern = re.compile(r"<.*?>")
        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.email_pattern = re.compile(r"\S+@\S+")

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, text_series):

        if not isinstance(text_series, pd.Series):
            raise TypeError("Input must be a pandas Series.")

        return text_series.fillna("").apply(self.clean_text)

    def clean_text(self, text):

        text = unicodedata.normalize(
            "NFKD",
            str(text)
        ).lower()

        text = self.html_pattern.sub(" ", text)
        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)

        # Preserve meaning
        text = text.replace("&", " and ")
        text = text.replace("-", " ")

        # Keep letters and numbers
        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        KEEP_TOKENS = {
            "3d",
            "4k",
            "vr",
            "ar",
            "ai",
            "usb",
            "ios",
            "android",
            "mp3",
            "bluetooth",
        }

        tokens = []

        for word in text.split():

            if word in self.stop_words:
                continue

            if len(word) <= 1:
                continue

            # Keep important technology tokens
            if word in KEEP_TOKENS:
                tokens.append(word)
                continue

            # Remove pure numbers
            if word.isdigit():
                continue

            # Remove ordinal numbers like 1st, 2nd, 10th
            if re.fullmatch(r"\d+(st|nd|rd|th)", word):
                continue

            lemma = self.lemmatizer.lemmatize(word)

            tokens.append(lemma)

        return " ".join(tokens)