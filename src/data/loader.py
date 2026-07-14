from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Loads and performs initial cleaning on the Kickstarter dataset.
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def load_data(self):

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.filepath}"
            )

        df = pd.read_csv(self.filepath)

        print("=" * 60)
        print("DATASET LOADED")
        print("=" * 60)
        print(f"Shape : {df.shape}")

        return df

    def initial_cleaning(self, df):

        df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_"))

        duplicates = df.duplicated().sum()

        print(f"\nDuplicate Rows : {duplicates}")

        df = df.drop_duplicates()

        df = df.dropna(how="all")

        df = df.dropna(axis=1, how="all")

        for col in df.columns:

            if (
                "date" in col or "deadline" in col or "launched" in col):

                try:
                    df[col] = pd.to_datetime(df[col])

                except Exception:
                    pass

        print("\nCleaning Complete")

        print(f"New Shape : {df.shape}")

        return df

    def save_clean_data(self, df, save_path):

        save_path = Path(save_path)

        save_path.parent.mkdir(parents=True,exist_ok=True)

        df.to_csv(save_path,index=False)

        print(f"\nSaved cleaned dataset to\n{save_path}")