from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDA:

    def __init__(
        self,
        figure_dir="reports/figures",
        table_dir="reports/tables",
    ):

        self.figure_dir = Path(figure_dir)
        self.table_dir = Path(table_dir)

        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    def dataset_overview(self, df):

        print("\n" + "=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print(f"Rows           : {df.shape[0]:,}")
        print(f"Columns        : {df.shape[1]}")
        print(
            f"Memory Usage   : "
            f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB"
        )

    # =====================================================
    # MISSING VALUES
    # =====================================================

    def missing_values(self, df):

        missing = pd.DataFrame({
            "Missing Count": df.isnull().sum(),
            "Missing %": (
                df.isnull().mean() * 100
            ).round(2)
        })

        missing.to_csv(
            self.table_dir / "missing_values.csv"
        )

        missing = missing[missing["Missing Count"] > 0]

        if len(missing):

            plt.figure(figsize=(8,5))

            sns.barplot(
                x=missing.index,
                y=missing["Missing Count"]
            )

            plt.xticks(rotation=45)

            plt.title("Missing Values")

            plt.tight_layout()

            plt.savefig(
                self.figure_dir /
                "missing_values.png"
            )

            plt.close()

    # =====================================================
    # NUMERICAL ANALYSIS
    # =====================================================

    def numerical_analysis(self, df):

        numeric = df.select_dtypes(
            include="number"
        )

        numeric.describe().T.to_csv(
            self.table_dir /
            "numerical_summary.csv"
        )

        numeric.skew().to_csv(
            self.table_dir /
            "skewness.csv"
        )

        # Histograms

        numeric.hist(
            figsize=(15,10),
            bins=30
        )

        plt.tight_layout()

        plt.savefig(
            self.figure_dir /
            "numerical_histograms.png"
        )

        plt.close()

        # Boxplots

        fig, axes = plt.subplots(
            len(numeric.columns),
            1,
            figsize=(8,4*len(numeric.columns))
        )

        if len(numeric.columns) == 1:
            axes = [axes]

        for ax, col in zip(
            axes,
            numeric.columns
        ):

            sns.boxplot(
                x=df[col],
                ax=ax
            )

            ax.set_title(col)

        plt.tight_layout()

        plt.savefig(
            self.figure_dir /
            "boxplots.png"
        )

        plt.close()

    # =====================================================
    # CORRELATION
    # =====================================================

    def correlation_analysis(self, df):

        numeric = df.select_dtypes(
            include="number"
        )

        corr = numeric.corr()

        corr.to_csv(
            self.table_dir /
            "correlation_matrix.csv"
        )

        plt.figure(figsize=(10,8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(
            self.figure_dir /
            "correlation_heatmap.png"
        )

        plt.close()

    # =====================================================
    # CATEGORICAL
    # =====================================================

    def categorical_analysis(self, df):

        cat = df.select_dtypes(
            include=["object", "string"]
        )

        summary = pd.DataFrame({

            "Unique Values":
                cat.nunique(),

            "Most Frequent":
                cat.mode().iloc[0],

            "Frequency":
                cat.apply(
                    lambda x:
                    x.value_counts().iloc[0]
                )

        })

        summary.to_csv(
            self.table_dir /
            "categorical_summary.csv"
        )

        for col in cat.columns:

            plt.figure(figsize=(10,5))

            (
                df[col]
                .value_counts()
                .head(15)
                .plot(kind="bar")
            )

            plt.title(col)

            plt.tight_layout()

            plt.savefig(
                self.figure_dir /
                f"{col}.png"
            )

            plt.close()

    # =====================================================
    # DATETIME
    # =====================================================

    def datetime_analysis(self, df):

        if (
            "launched" in df.columns
            and
            "deadline" in df.columns
        ):

            duration = (
                df["deadline"] -
                df["launched"]
            ).dt.days

            duration.describe().to_csv(
                self.table_dir /
                "campaign_duration.csv"
            )

            plt.figure(figsize=(8,5))

            duration.hist(bins=40)

            plt.title(
                "Campaign Duration"
            )

            plt.tight_layout()

            plt.savefig(
                self.figure_dir /
                "campaign_duration.png"
            )

            plt.close()

    # =====================================================
    # TEXT
    # =====================================================

    def text_analysis(self, df):

        if "name" not in df.columns:
            return

        chars = (
            df["name"]
            .fillna("")
            .str.len()
        )

        words = (
            df["name"]
            .fillna("")
            .str.split()
            .str.len()
        )

        summary = pd.DataFrame({

            "Characters":
                chars.describe(),

            "Words":
                words.describe()

        })

        summary.to_csv(
            self.table_dir /
            "text_statistics.csv"
        )

        plt.figure(figsize=(8,5))

        chars.hist(bins=40)

        plt.title(
            "Campaign Title Length"
        )

        plt.tight_layout()

        plt.savefig(
            self.figure_dir /
            "campaign_title_length.png"
        )

        plt.close()

    # =====================================================
    # RUN
    # =====================================================

    def run(self, df):

        print("\nGenerating EDA Reports...")

        self.dataset_overview(df)

        self.missing_values(df)

        self.numerical_analysis(df)

        self.correlation_analysis(df)

        self.categorical_analysis(df)

        self.datetime_analysis(df)

        self.text_analysis(df)

        print("\nEDA Completed Successfully!")