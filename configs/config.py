RANDOM_STATE = 42
TARGET_COLUMN = "state"

POSITIVE_CLASS = "successful"

NEGATIVE_CLASS = "failed"

LEAKAGE_COLUMNS = [
    "pledged",
    "backers",
    "usd_pledged",
    "usd_pledged_real",
]

DROP_COLUMNS = [
    "id",
]

TFIDF_CONFIG = {

    "max_features": 5000,

    "ngram_range": (1, 2),

    "min_df": 10,

    "max_df": 0.9,

    "sublinear_tf": True,

    "strip_accents": "unicode"
}

TEST_SIZE = 0.20