from src.data.loader import DataLoader


DATA_PATH = "data/raw/ks-projects-201801.csv"

SAVE_PATH = "data/interim/cleaned_dataset.csv"


loader = DataLoader(DATA_PATH)

df = loader.load_data()

df = loader.initial_cleaning(df)

loader.save_clean_data(df, SAVE_PATH)

print("\n")

print(df.head())

print("\n")

print(df.info())