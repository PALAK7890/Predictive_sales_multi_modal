"""
Global Configuration
"""

# =====================================================
# RANDOM STATE
# =====================================================

RANDOM_STATE = 42

# =====================================================
# DATASET CONFIG
# =====================================================

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

# =====================================================
# TF-IDF
# =====================================================

TFIDF_CONFIG = {

    "max_features": 2500,

    "ngram_range": (1, 2),

    "min_df": 5,

    "max_df": 0.95,

    "sublinear_tf": True,

    "strip_accents": "unicode"
}

TEST_SIZE = 0.20