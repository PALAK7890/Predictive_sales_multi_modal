from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from configs.config import RANDOM_STATE, TEST_SIZE


class ModelTrainer:

    def __init__(self):
        self.model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    def split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    def fit(self, X_train, y_train):
        print("\nTraining Logistic Regression...")

        self.model.fit(X_train, y_train)
        self.save()

        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self):
        model_dir = Path("models/classifiers")
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.model,
            model_dir / "logistic_regression.pkl",
        )