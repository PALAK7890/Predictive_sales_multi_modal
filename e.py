import joblib

preprocessor = joblib.load("models/encoders/tabular_preprocessor.pkl")

print(preprocessor.feature_names_in_)