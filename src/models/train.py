from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

from configs.config import RANDOM_STATE, TEST_SIZE


class ModelTrainer:

    def __init__(self, model_name="logistic"):

        self.model_name = model_name

        if model_name == "logistic":

            self.model = LogisticRegression(
                solver="liblinear",
                random_state=RANDOM_STATE,
                max_iter=1000,
            )

        elif model_name == "svm":

            self.model = LinearSVC(
                random_state=RANDOM_STATE,
                max_iter=5000,
            )

        else:

            raise ValueError(
                f"Unknown model: {model_name}"
            )

    def split(self, X, y):

        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    def tune(self, X_train, y_train):

        print(f"\nRunning GridSearchCV ({self.model_name})...\n")

        if self.model_name == "logistic":

            param_grid = {
                "C": [0.1, 1, 10],
                "solver": ["liblinear"],
                "class_weight": [None, "balanced"],
            }

        elif self.model_name == "svm":

            param_grid = {
                "C": [0.1, 1, 10],
                "class_weight": [None, "balanced"],
            }

        else:

            raise ValueError(
                f"No parameter grid defined for '{self.model_name}'"
            )

        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_

        print("\nBest Parameters")
        print(grid.best_params_)

        print(f"\nBest CV F1 Score : {grid.best_score_:.4f}")

        return self.model

    def fit(self, X_train, y_train):

        model = self.tune(X_train, y_train)

        print(f"\nTraining Best {self.model_name.upper()}...\n")

        model.fit(X_train, y_train)

        self.save()

        return model

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        raise AttributeError(
            f"{self.model_name} does not support predict_proba()."
        )

    def save(self):

        model_dir = Path("models/classifiers")
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{self.model_name}.pkl"
        )

        joblib.dump(
            self.model,
            model_dir / filename,
        )