from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.visualization.eda import EDA
from src.preprocessing.text_cleaner import TextCleaner
from src.vectorization.tfidf import TFIDFVectorizer

class Pipeline:

    def __init__(self):

        self.data_path = "data/raw/ks-projects-201801.csv"

        self.save_path = "data/interim/cleaned_dataset.csv"

        self.loader = DataLoader(self.data_path)

        self.validator = DataValidator()

        self.eda = EDA()
        self.cleaner = TextCleaner()
        self.vectorizer = TFIDFVectorizer()

    # =====================================================
    # DATA INGESTION
    # =====================================================

    def load_data(self):

        df = self.loader.load_data()

        df = self.loader.initial_cleaning(df)

        self.loader.save_clean_data(
            df,
            self.save_path
        )

        return df

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self, df):

        self.validator.validate(df)

    # =====================================================
    # EDA
    # =====================================================

    def perform_eda(self, df):

        self.eda.run(df)

    # =====================================================
    # PIPELINE
    # =====================================================

    def preprocess_text(self, df):

        print("\nCleaning text...")

        df["clean_text"] = self.cleaner.fit_transform(
            df["name"]
        )

        print("\nSample Cleaned Text\n")
        print(
            df[
                ["name", "clean_text"]
            ].head(10)
        )

        print("\nMissing cleaned text:")
        print(df["clean_text"].isna().sum())

        df.to_csv(
            "data/interim/cleaned_text_dataset.csv",
            index=False
        )

        print(
            "\nSaved cleaned dataset to "
            "data/interim/cleaned_text_dataset.csv"
        )

        return df
    # =====================================================
# TF-IDF
# =====================================================

    def vectorize_text(self, df):

        print("\nGenerating TF-IDF Matrix...")

        X_text = self.vectorizer.fit_transform(

            df["clean_text"]

        )

        self.vectorizer.save()

        self.vectorizer.save_vocabulary()

        print(

            f"Vocabulary Size : "

            f"{self.vectorizer.vocabulary_size()}"

        )

        print(

            f"Sparse Matrix Shape : "

            f"{X_text.shape}"

        )

        return X_text

    def run(self):

        print("\nStarting Pipeline...\n")

        df = self.load_data()

        self.validate(df)

        self.perform_eda(df)

        df = self.preprocess_text(df)
        X_text = self.vectorize_text(df)

        print("\nPipeline Completed Successfully!")