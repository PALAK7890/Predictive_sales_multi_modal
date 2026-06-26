from src.data.loader import DataLoader
from src.data.validator import DataValidator


DATA_PATH = "data/raw/ks-projects-201801.csv"
SAVE_PATH = "data/interim/cleaned_dataset.csv"

loader = DataLoader(DATA_PATH)

df = loader.load_data()

df = loader.initial_cleaning(df)

loader.save_clean_data(df, SAVE_PATH)

validator = DataValidator()

validator.validate(df)