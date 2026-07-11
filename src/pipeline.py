from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.visualization.eda import EDA
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tabular_preprocessor import TabularPreprocessor
from src.vectorization.tfidf import TFIDFVectorizer
from src.fusion.feature_fusion import FeatureFusion
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator


class Pipeline:

    def __init__(self):
        self.data_path = "data/raw/ks-projects-201801.csv"
        self.save_path = "data/interim/cleaned_dataset.csv"

        self.loader = DataLoader(self.data_path)
        self.validator = DataValidator()
        self.eda = EDA()
        self.cleaner = TextCleaner()
        self.vectorizer = TFIDFVectorizer()
        self.tabular = TabularPreprocessor()
        self.fusion = FeatureFusion()
        self.trainer = ModelTrainer("svm")
        self.evaluator = ModelEvaluator()

    def load_data(self):
        df = self.loader.load_data()
        df = self.loader.initial_cleaning(df)
        self.loader.save_clean_data(df, self.save_path)
        return df

    def validate(self, df):
        self.validator.validate(df)

    def perform_eda(self, df):
        self.eda.run(df)

    def preprocess_text(self, df):
        print("\nCleaning text...")

        df["clean_text"] = self.cleaner.fit_transform(df["name"])

        print("\nSample Cleaned Text")
        print(df[["name", "clean_text"]].head(10))

        print(f"\nMissing cleaned text: {df['clean_text'].isna().sum()}")

        output_path = "data/interim/cleaned_text_dataset.csv"
        df.to_csv(output_path, index=False)

        print(f"\nSaved cleaned dataset to {output_path}")

        return df

    def vectorize_text(self, df):
        print("\nGenerating TF-IDF Matrix...")

        X_text = self.vectorizer.fit_transform(df["clean_text"])

        self.vectorizer.save()
        self.vectorizer.save_vocabulary()

        print(f"Vocabulary Size : {self.vectorizer.vocabulary_size()}")
        print(f"Sparse Matrix Shape : {X_text.shape}")

        return X_text

    def preprocess_tabular(self, df):
        print("\nProcessing tabular features...")

        X_tabular, y = self.tabular.fit_transform(df)

        self.tabular.summary()
        self.tabular.save()

        print(f"\nTabular Matrix Shape : {X_tabular.shape}")
        print(f"Target Shape         : {y.shape}")

        return X_tabular, y

    def fuse_features(self, X_text, X_tabular, y):
        print("\nFusing features...")
        return self.fusion.fit_transform(X_text, X_tabular, y)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = self.trainer.split(X, y)

        self.trainer.fit(X_train, y_train)
        self.trainer.save()

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X_test, y_test):
        return self.evaluator.evaluate(
            self.trainer.model,
            X_test,
            y_test,
        )

    def run(self):
        print("\nStarting pipeline...\n")

        df = self.load_data()
        df = self.tabular.filter_final_campaigns(df)

        self.validate(df)
        self.perform_eda(df)

        df = self.preprocess_text(df)

        X_text = self.vectorize_text(df)
        X_tabular, y = self.preprocess_tabular(df)

        X, y = self.fuse_features(X_text, X_tabular, y)

        _, X_test, _, y_test = self.train_model(X, y)

        self.evaluate_model(X_test, y_test)

        print("\nPipeline completed successfully!")

        return X, y