from pathlib import Path

import joblib
import numpy as np

from scipy.sparse import (
    hstack,
    save_npz,
)


class FeatureFusion:
    """
    Combines text features and tabular features
    into one sparse feature matrix.
    """

    def __init__(self):

        self.output_dir = Path("data/processed")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self, X_text, X_tabular, y):

        if X_text.shape[0] != X_tabular.shape[0]:

            raise ValueError(

                f"Row mismatch:\n"

                f"Text     : {X_text.shape[0]}\n"

                f"Tabular  : {X_tabular.shape[0]}"

            )

        if len(y) != X_text.shape[0]:

            raise ValueError(

                f"Target mismatch:\n"

                f"Target : {len(y)}\n"

                f"Rows   : {X_text.shape[0]}"

            )

    # =====================================================
    # FEATURE FUSION
    # =====================================================

    def fuse(self, X_text, X_tabular):

        return hstack(

            [

                X_text,

                X_tabular,

            ],

            format="csr",

        )

    # =====================================================
    # SAVE
    # =====================================================

    def save(self, X, y):

        save_npz(

            self.output_dir / "X_fused.npz",

            X

        )

        np.save(

            self.output_dir / "y.npy",

            y

        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self, X):

        print("\n" + "=" * 60)

        print("FEATURE FUSION")

        print("=" * 60)

        print(f"Samples      : {X.shape[0]:,}")

        print(f"Features     : {X.shape[1]:,}")

        print(f"Non-Zero     : {X.nnz:,}")

        density = (

            X.nnz /

            (X.shape[0] * X.shape[1])

        )

        print(f"Density      : {density:.6f}")

        print(f"Sparsity     : {(1-density)*100:.2f}%")



    def fit_transform(
        self,
        X_text,
        X_tabular,
        y,
    ):

        self.validate(
            X_text,
            X_tabular,
            y,
        )

        X = self.fuse(
            X_text,
            X_tabular,
        )

        self.save(
            X,
            y,
        )

        self.summary(X)

        return X, y