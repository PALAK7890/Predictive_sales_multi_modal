from pathlib import Path
import pandas as pd


class DataValidator:
    """
    Performs dataset validation and generates summary reports.
    """

    def __init__(self, output_dir="reports/tables"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Dataset Shape
    # --------------------------------------------------------

    def dataset_shape(self, df):

        rows, cols = df.shape

        print("\n" + "=" * 60)
        print("DATASET SHAPE")
        print("=" * 60)

        print(f"Rows    : {rows:,}")
        print(f"Columns : {cols}")

    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------

    def missing_values(self, df):

        missing = pd.DataFrame({
            "Missing_Count": df.isnull().sum(),
            "Missing_Percentage":
                (df.isnull().sum() / len(df) * 100).round(2)
        })

        missing = missing.sort_values(
            by="Missing_Count",
            ascending=False
        )

        print("\n" + "=" * 60)
        print("MISSING VALUES")
        print("=" * 60)

        print(missing)

        missing.to_csv(
            self.output_dir / "missing_values.csv"
        )

    # --------------------------------------------------------
    # Duplicate Rows
    # --------------------------------------------------------

    def duplicate_rows(self, df):

        duplicates = df.duplicated().sum()

        print("\n" + "=" * 60)
        print("DUPLICATE ROWS")
        print("=" * 60)

        print(f"Duplicate Rows : {duplicates}")

    # --------------------------------------------------------
    # Data Types
    # --------------------------------------------------------

    def data_types(self, df):

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "DataType": df.dtypes.astype(str)
        })

        print("\n" + "=" * 60)
        print("DATA TYPES")
        print("=" * 60)

        print(dtype_df)

        dtype_df.to_csv(
            self.output_dir / "data_types.csv",
            index=False
        )

    # --------------------------------------------------------
    # Numerical Summary
    # --------------------------------------------------------

    def numerical_summary(self, df):

        numerical = df.select_dtypes(
            include=["number"]
        )

        summary = numerical.describe().T

        print("\n" + "=" * 60)
        print("NUMERICAL SUMMARY")
        print("=" * 60)

        print(summary)

        summary.to_csv(
            self.output_dir / "numerical_summary.csv"
        )

    # --------------------------------------------------------
    # Categorical Summary
    # --------------------------------------------------------

    def categorical_summary(self, df):

        categorical = df.select_dtypes(
            include=["object", "string", "category"]
        )

        summary = pd.DataFrame({

            "Unique_Values": categorical.nunique(),

            "Most_Frequent":
                categorical.mode().iloc[0],

            "Frequency":
                categorical.apply(
                    lambda x: x.value_counts().iloc[0]
                )

        })

        print("\n" + "=" * 60)
        print("CATEGORICAL SUMMARY")
        print("=" * 60)

        print(summary)

        summary.to_csv(
            self.output_dir / "categorical_summary.csv"
        )

    # --------------------------------------------------------
    # Run All
    # --------------------------------------------------------

    def validate(self, df):

        self.dataset_shape(df)

        self.missing_values(df)

        self.duplicate_rows(df)

        self.data_types(df)

        self.numerical_summary(df)

        self.categorical_summary(df)

        print("\nValidation Completed Successfully.")