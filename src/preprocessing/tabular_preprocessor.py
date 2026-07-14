import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import (
    TARGET_COLUMN,
    POSITIVE_CLASS,
    NEGATIVE_CLASS,
    LEAKAGE_COLUMNS,
    DROP_COLUMNS,
)


class TabularPreprocessor:

    def __init__(self):
        self.preprocessor = None
        self.numeric_columns = []
        self.categorical_columns = []

    def clean_target(self, df):

        df = df[df[TARGET_COLUMN].isin([POSITIVE_CLASS, NEGATIVE_CLASS])].copy()

        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
            POSITIVE_CLASS: 1,
            NEGATIVE_CLASS: 0,
        })

        return df

    def filter_final_campaigns(self, df):

        return df[
            df[TARGET_COLUMN].isin(
                [POSITIVE_CLASS, NEGATIVE_CLASS]
            )
        ].copy()

    def remove_leakage(self, df):

        columns = [
            col
            for col in LEAKAGE_COLUMNS
            if col in df.columns
        ]

        return df.drop(columns=columns)

    def drop_unused(self, df):

        columns = [
            col
            for col in DROP_COLUMNS
            if col in df.columns
        ]

        return df.drop(columns=columns)

    def engineer_features(self, df):

        df = df.copy()

        df["goal_log"] = np.log1p(df["goal"])
        df["usd_goal_real_log"] = np.log1p(df["usd_goal_real"])
        df["title_length"] = df["name"].str.len()
        df["title_word_count"] = df["name"].str.split().str.len()
        

        if {"launched", "deadline"}.issubset(df.columns):

            df["campaign_duration"] = (
                df["deadline"] - df["launched"]
            ).dt.days

            df["launch_year"] = df["launched"].dt.year
            df["launch_month"] = df["launched"].dt.month
            df["launch_day"] = df["launched"].dt.day
            df["launch_weekday"] = df["launched"].dt.weekday
            df["launch_quarter"] = df["launched"].dt.quarter

            df.drop(
                columns=["launched", "deadline"],
                inplace=True,
            )
        df["goal_per_day"] = df["goal_log"] / (df["campaign_duration"] + 1)

        return df

    def detect_columns(self, X):

        self.numeric_columns = list(
            X.select_dtypes(include="number").columns
        )

        self.categorical_columns = list(
            X.select_dtypes(
                include=["object", "category", "string"]
            ).columns
        )

        remove_columns = {"name", "clean_text"}

        self.categorical_columns = [
            col
            for col in self.categorical_columns
            if col not in remove_columns
        ]

    def build_preprocessor(self):

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        self.preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, self.numeric_columns),
            ("categorical", categorical_pipeline, self.categorical_columns),
        ])
    def fit(self, df):

        df = self.clean_target(df)
        df = self.remove_leakage(df)
        df = self.drop_unused(df)
        df = self.engineer_features(df)

        y = df[TARGET_COLUMN]
        X = df.drop(columns=[TARGET_COLUMN])

        self.detect_columns(X)
        self.build_preprocessor()
        self.preprocessor.fit(X)

        return self

    def transform(self, df):

        df = self.clean_target(df)
        df = self.remove_leakage(df)
        df = self.drop_unused(df)
        df = self.engineer_features(df)

        y = df[TARGET_COLUMN]
        X = df.drop(columns=[TARGET_COLUMN])

        return self.preprocessor.transform(X), y

    def fit_transform(self, df):

        self.fit(df)
        return self.transform(df)

    def save(self):

        import joblib
        from pathlib import Path

        save_dir = Path("models/encoders")
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.preprocessor,
            save_dir / "tabular_preprocessor.pkl",
        )

    def summary(self):

        print("\n" + "=" * 60)
        print("TABULAR PREPROCESSOR")
        print("=" * 60)

        print(f"Numeric Columns    : {len(self.numeric_columns)}")
        print(f"Categorical Columns: {len(self.categorical_columns)}")

        print("\nNumeric Features")
        for col in self.numeric_columns:
            print(f"  • {col}")

        print("\nCategorical Features")
        for col in self.categorical_columns:
            print(f"  • {col}")