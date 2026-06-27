"""
Global project configuration.
Modify values here instead of changing source code.
"""

# =====================================================
# RANDOMNESS
# =====================================================

RANDOM_STATE = 42

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

# =====================================================
# TRAIN / TEST
# =====================================================

TEST_SIZE = 0.20