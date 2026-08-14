"""
XGBoost Tabular Preprocessor
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class XGBPreprocessor:

    def __init__(self):

        self.preprocessor = None

    def feature_engineering(self, df: pd.DataFrame):

        df = df.copy()

        df["launched"] = pd.to_datetime(df["launched"])
        df["deadline"] = pd.to_datetime(df["deadline"])

        df["campaign_duration"] = (
            df["deadline"] - df["launched"]
        ).dt.days

        df["launch_year"] = df["launched"].dt.year
        df["launch_month"] = df["launched"].dt.month
        df["launch_day"] = df["launched"].dt.day
        df["launch_weekday"] = df["launched"].dt.weekday
        df["launch_quarter"] = df["launched"].dt.quarter

        df["goal_log"] = np.log1p(df["goal"])
        df["usd_goal_real_log"] = np.log1p(df["usd_goal_real"])

        df["title_length"] = (
            df["name"]
            .fillna("")
            .str.len()
        )

        df["title_word_count"] = (
            df["name"]
            .fillna("")
            .str.split()
            .str.len()
        )

        df["goal_per_day"] = (
            df["usd_goal_real"] /
            df["campaign_duration"].replace(0, 1)
        )

        return df

    def fit(self, df):

        df = self.feature_engineering(df)

        numeric_features = [
            "goal",
            "goal_log",
            "usd_goal_real",
            "usd_goal_real_log",
            "campaign_duration",
            "launch_year",
            "launch_month",
            "launch_day",
            "launch_weekday",
            "launch_quarter",
            "title_length",
            "title_word_count",
            "goal_per_day",
            "backers",
        ]

        categorical_features = [
            "country",
            "currency",
            "category",
            "main_category",
        ]

        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        self.preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features,
                ),
            ]
        )

        self.preprocessor.fit(df)

        return self

    def transform(self, df):

        df = self.feature_engineering(df)

        return self.preprocessor.transform(df)

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)

    def save(
        self,
        path="artifacts/preprocessors/xgb_preprocessor.pkl",
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.preprocessor,
            path,
        )

    def load(
        self,
        path="artifacts/preprocessors/xgb_preprocessor.pkl",
    ):

        self.preprocessor = joblib.load(path)

        return self


def main():

    print("=" * 60)
    print("Building XGBoost Preprocessor")
    print("=" * 60)

    df = pd.read_csv(
        "data/interim/cleaned_text_dataset.csv"
    )

    processor = XGBPreprocessor()

    X = processor.fit_transform(df)

    processor.save()

    print("\nPreprocessor Saved")

    print("Output Shape:", X.shape)


if __name__ == "__main__":
    main()