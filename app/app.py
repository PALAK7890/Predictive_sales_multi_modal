import os
import sys
import re
import unicodedata
import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse import hstack

# ==========================================================
# PATH RESOLUTION & ENVIRONMENT SETUP
# ==========================================================

# Resolve project root dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set NLTK data directory locally to prevent sandbox blocks
import nltk
nltk.data.path = [os.path.join(project_root, 'nltk_data')]

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# TEXT CLEANER CLASS REPLICATION
# ==========================================================

class TextCleaner:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            # Fallback in case NLTK data is not loaded yet
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
        self.html_pattern = re.compile(r"<.*?>")
        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.email_pattern = re.compile(r"\S+@\S+")

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, text_series):
        if not isinstance(text_series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
        return text_series.fillna("").apply(self.clean_text)

    def clean_text(self, text):
        text = unicodedata.normalize("NFKD", str(text)).lower()
        text = self.html_pattern.sub(" ", text)
        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)

        # Preserve meaning
        text = text.replace("&", " and ")
        text = text.replace("-", " ")

        # Keep letters and numbers
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        KEEP_TOKENS = {
            "3d", "4k", "vr", "ar", "ai", "usb", "ios", "android", "mp3", "bluetooth"
        }

        tokens = []
        for word in text.split():
            if word in self.stop_words:
                continue
            if len(word) <= 1:
                continue
            if word in KEEP_TOKENS:
                tokens.append(word)
                continue
            if word.isdigit():
                continue
            if re.fullmatch(r"\d+(st|nd|rd|th)", word):
                continue

            lemma = self.lemmatizer.lemmatize(word)
            tokens.append(lemma)

        return " ".join(tokens)

# ==========================================================
# RESOURCE LOADER
# ==========================================================

@st.cache_resource
def load_ml_resources():
    classifier_path = os.path.join(project_root, "models/classifiers/logistic.pkl")
    vectorizer_path = os.path.join(project_root, "models/vectorizers/tfidf.pkl")
    preprocessor_path = os.path.join(project_root, "models/encoders/tabular_preprocessor.pkl")
    
    model = joblib.load(classifier_path)
    vectorizer = joblib.load(vectorizer_path)
    tabular = joblib.load(preprocessor_path)
    
    # Load global feature importance table if available
    importance_path = os.path.join(project_root, "reports/tables/feature_importance.csv")
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
    else:
        importance_df = None
        
    return model, vectorizer, tabular, importance_df

# Load assets
try:
    model, vectorizer, tabular, global_importance_df = load_ml_resources()
    cleaner = TextCleaner()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Make sure the models/ folder contains the required .pkl files.")
    st.stop()

# ==========================================================
# CATEGORY MAPPING & UTILITIES
# ==========================================================

category_mapping = {
    'Art': ['Art', 'Ceramics', 'Conceptual Art', 'Digital Art', 'Illustration', 'Installations', 'Mixed Media', 'Painting', 'Performance Art', 'Public Art', 'Sculpture', 'Textiles', 'Video Art'],
    'Comics': ['Anthologies', 'Comic Books', 'Comics', 'Events', 'Graphic Novels', 'Webcomics'],
    'Crafts': ['Candles', 'Crafts', 'Crochet', 'DIY', 'Embroidery', 'Glass', 'Knitting', 'Letterpress', 'Pottery', 'Printing', 'Quilts', 'Stationery', 'Taxidermy', 'Weaving', 'Woodworking'],
    'Dance': ['Dance', 'Performances', 'Residencies', 'Spaces', 'Workshops'],
    'Design': ['Architecture', 'Civic Design', 'Design', 'Graphic Design', 'Interactive Design', 'Product Design', 'Typography'],
    'Fashion': ['Accessories', 'Apparel', 'Childrenswear', 'Couture', 'Fashion', 'Footwear', 'Jewelry', 'Pet Fashion', 'Ready-to-wear'],
    'Film & Video': ['Action', 'Animation', 'Comedy', 'Documentary', 'Drama', 'Experimental', 'Family', 'Fantasy', 'Festivals', 'Film & Video', 'Horror', 'Movie Theaters', 'Music Videos', 'Narrative Film', 'Romance', 'Science Fiction', 'Shorts', 'Television', 'Thrillers', 'Webseries'],
    'Food': ['Bacon', 'Community Gardens', 'Cookbooks', 'Drinks', 'Events', "Farmer's Markets", 'Farms', 'Food', 'Food Trucks', 'Restaurants', 'Small Batch', 'Spaces', 'Vegan'],
    'Games': ['Games', 'Gaming Hardware', 'Live Games', 'Mobile Games', 'Playing Cards', 'Puzzles', 'Tabletop Games', 'Video Games'],
    'Journalism': ['Audio', 'Journalism', 'Photo', 'Print', 'Video', 'Web'],
    'Music': ['Blues', 'Chiptune', 'Classical Music', 'Comedy', 'Country & Folk', 'Electronic Music', 'Faith', 'Hip-Hop', 'Indie Rock', 'Jazz', 'Kids', 'Latin', 'Metal', 'Music', 'Pop', 'Punk', 'R&B', 'Rock', 'World Music'],
    'Photography': ['Animals', 'Fine Art', 'Nature', 'People', 'Photobooks', 'Photography', 'Places'],
    'Publishing': ['Academic', 'Anthologies', 'Art Books', 'Calendars', "Children's Books", 'Comedy', 'Fiction', 'Letterpress', 'Literary Journals', 'Literary Spaces', 'Nonfiction', 'Periodicals', 'Poetry', 'Publishing', 'Radio & Podcasts', 'Translations', 'Young Adult', 'Zines'],
    'Technology': ['3D Printing', 'Apps', 'Camera Equipment', 'DIY Electronics', 'Fabrication Tools', 'Flight', 'Gadgets', 'Hardware', 'Makerspaces', 'Robots', 'Software', 'Sound', 'Space Exploration', 'Technology', 'Wearables', 'Web'],
    'Theater': ['Comedy', 'Experimental', 'Festivals', 'Immersive', 'Musical', 'Plays', 'Spaces', 'Theater']
}

exchange_rates = {
    'AUD': 0.7953215892956982,
    'CAD': 0.8171686780860878,
    'CHF': 1.0157961494700183,
    'DKK': 0.15061655617524705,
    'EUR': 1.1240804688182744,
    'GBP': 1.4888240936688848,
    'HKD': 0.12832059569627405,
    'JPY': 0.008885902508955957,
    'MXN': 0.05265503053829873,
    'NOK': 0.12337482792286146,
    'NZD': 0.73823013375278,
    'SEK': 0.1186420318380655,
    'SGD': 0.7253183231321773,
    'USD': 1.0
}

country_names = {
    'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada', 'AU': 'Australia',
    'NZ': 'New Zealand', 'SG': 'Singapore', 'CH': 'Switzerland', 'SE': 'Sweden',
    'NO': 'Norway', 'DK': 'Denmark', 'HK': 'Hong Kong', 'MX': 'Mexico', 'JP': 'Japan',
    'AT': 'Austria', 'BE': 'Belgium', 'DE': 'Germany', 'ES': 'Spain', 'FR': 'France',
    'IE': 'Ireland', 'IT': 'Italy', 'LU': 'Luxembourg', 'NL': 'Netherlands',
    'N,0"': 'Other / Undefined'
}

currency_names = {
    'USD': 'US Dollar ($)', 'GBP': 'British Pound (£)', 'EUR': 'Euro (€)',
    'CAD': 'Canadian Dollar (C$)', 'AUD': 'Australian Dollar (A$)',
    'NZD': 'New Zealand Dollar (NZ$)', 'SGD': 'Singapore Dollar (S$)',
    'CHF': 'Swiss Franc (CHF)', 'SEK': 'Swedish Krona (kr)',
    'NOK': 'Norwegian Krone (kr)', 'DKK': 'Danish Krone (kr)',
    'HKD': 'Hong Kong Dollar (HK$)', 'MXN': 'Mexican Peso (MXN$)',
    'JPY': 'Japanese Yen (¥)'
}

country_to_currency_default = {
    'US': 'USD', 'GB': 'GBP', 'CA': 'CAD', 'AU': 'AUD', 'NZ': 'NZD',
    'SG': 'SGD', 'CH': 'CHF', 'SE': 'SEK', 'NO': 'NOK', 'DK': 'DKK',
    'HK': 'HKD', 'MX': 'MXN', 'JP': 'JPY',
    'AT': 'EUR', 'BE': 'EUR', 'DE': 'EUR', 'ES': 'EUR', 'FR': 'EUR',
    'IE': 'EUR', 'IT': 'EUR', 'LU': 'EUR', 'NL': 'EUR'
}

# ==========================================================
# CUSTOM CSS (APPLE SAAS THEME)
# ==========================================================

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    /* Global Background and Typography */
    [data-testid="stAppViewContainer"] {
        background-color: #f6f6f9;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(246, 246, 249, 0.8);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar Layout Customization */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }
    
    /* Navigation Sidebar Labels */
    [data-testid="stSidebar"] label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1d1d1f;
    }
    
    /* Modern Card Layouts */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.04);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.015);
        transition: all 0.2s ease-in-out;
        margin-bottom: 12px;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.03);
        border-color: rgba(0, 113, 227, 0.15);
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #86868b;
        margin-bottom: 4px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1d1d1f;
        margin: 0;
    }
    
    .saas-card {
        background-color: #ffffff;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid rgba(0, 0, 0, 0.04);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.015);
        margin-bottom: 24px;
    }
    
    /* Predict Button Custom Styling */
    div.stButton > button {
        background-color: #0071e3 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.2) !important;
    }
    
    div.stButton > button:hover {
        background-color: #0077ed !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0, 113, 227, 0.3) !important;
    }
    
    div.stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Result Header Callouts */
    .result-callout {
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .result-success {
        background-color: rgba(52, 199, 89, 0.08);
        border: 1px solid rgba(52, 199, 89, 0.2);
        color: #1d1d1f;
    }
    
    .result-failure {
        background-color: rgba(255, 59, 48, 0.08);
        border: 1px solid rgba(255, 59, 48, 0.2);
        color: #1d1d1f;
    }
    
    /* Titles and Typography adjustments */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.6px !important;
        color: #1d1d1f !important;
        font-size: 2.2rem !important;
    }
    
    h2 {
        font-weight: 700 !important;
        letter-spacing: -0.4px !important;
        color: #1d1d1f !important;
        font-size: 1.6rem !important;
    }
    
    h3 {
        font-weight: 600 !important;
        color: #1d1d1f !important;
        font-size: 1.25rem !important;
        margin-bottom: 8px !important;
    }
    
    .about-section {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background-color: #f2f2f7;
        color: #1d1d1f;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:
    st.markdown("### 📊 SaaS Platform")
    page = st.radio(
        "Navigation",
        ["Prediction", "Dashboard", "About"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Model Configuration")
    st.info("⚡ **Logistic Regression** classifier is active.")
    st.caption("Fitted on Kickstarter 2018 datasets with TF-IDF fusion pipeline.")

# ==========================================================
# APP CONTAINER & KPI HEADER
# ==========================================================

# High-level metrics header (rendered on Prediction and Dashboard)
if page in ["Prediction", "Dashboard"]:
    st.title("🎯 Kickstarter Campaign Success Prediction")
    st.caption("A production-ready SaaS dashboard for analyzing and predicting funding campaign success rates.")
    
    # KPI Row
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">68.57%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols[1]:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value">64.19%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols[2]:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">ROC AUC</div>
                <div class="metric-value">75.61%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols[3]:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Vocabulary Size</div>
                <div class="metric-value">5,000</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# PAGE 1 & 2: PREDICTION INTERFACE
# ==========================================================

if page == "Prediction":
    # Form layout
    st.markdown("### 📝 Campaign Configuration Form")
    
    with st.container():
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        
        # Row 1: Title input
        title = st.text_input(
            "Project Title",
            placeholder="e.g. A revolutionary 3D printer for tabletop miniature gaming",
            help="The public title of your Kickstarter campaign. Used for Text Vectorization."
        )
        
        # Row 2: Category controls
        col_main, col_sub = st.columns(2)
        with col_main:
            main_category_sel = st.selectbox(
                "Main Category",
                options=list(category_mapping.keys()),
                index=8  # Default to Games
            )
        with col_sub:
            sub_options = category_mapping[main_category_sel]
            category_sel = st.selectbox(
                "Subcategory",
                options=sub_options,
                index=sub_options.index("Tabletop Games") if "Tabletop Games" in sub_options else 0
            )
            
        # Row 3: Geographical & Financial controls
        col_country, col_curr, col_goal = st.columns(3)
        with col_country:
            country_code = st.selectbox(
                "Target Country",
                options=sorted(list(country_names.keys())),
                format_func=lambda x: f"{x} - {country_names[x]}",
                index=sorted(list(country_names.keys())).index("US")
            )
        with col_curr:
            # Set default currency based on selected country
            default_currency = country_to_currency_default.get(country_code, "USD")
            curr_options = sorted(list(currency_names.keys()))
            currency_code = st.selectbox(
                "Campaign Currency",
                options=curr_options,
                format_func=lambda x: f"{x} ({currency_names[x]})",
                index=curr_options.index(default_currency)
            )
        with col_goal:
            goal_amount = st.number_input(
                "Goal Amount (Local Currency)",
                min_value=1.0,
                max_value=1e9,
                value=5000.0,
                step=100.0,
                format="%.2f"
            )
            
        # Row 4: Temporal controls
        col_date, col_dur = st.columns(2)
        with col_date:
            launch_date_val = st.date_input(
                "Simulated Launch Date",
                value=datetime.date.today(),
                help="The date when the campaign is launched."
            )
        with col_dur:
            duration_val = st.number_input(
                "Campaign Duration (Days)",
                min_value=1,
                max_value=120,
                value=30,
                step=1,
                help="Duration of the campaign in days."
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        predict_click = st.button("🚀 Predict Success Probability")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Execute prediction when clicked
    if predict_click:
        if not title.strip():
            st.warning("⚠️ Please enter a project title to run predictions.")
        else:
            with st.spinner("Processing NLP features & Running classification..."):
                # 1. Clean Title using TextCleaner
                clean_title = cleaner.clean_text(title)
                
                # 2. Re-create exchange rate and USD goals
                usd_goal_real = goal_amount * exchange_rates.get(currency_code, 1.0)
                
                # 3. Re-create engineered features
                goal_log = np.log1p(goal_amount)
                usd_goal_real_log = np.log1p(usd_goal_real)
                title_length = len(title)
                title_word_count = len(title.split())
                campaign_duration = int(duration_val)
                
                launch_pd = pd.Timestamp(launch_date_val)
                launch_year = launch_pd.year
                launch_month = launch_pd.month
                launch_day = launch_pd.day
                launch_weekday = launch_pd.weekday() # 0-indexed (Monday=0)
                launch_quarter = launch_pd.quarter
                
                goal_per_day = goal_log / (campaign_duration + 1)
                
                # 4. Construct input DataFrame matching ColumnTransformer order exactly
                input_df = pd.DataFrame([{
                    "name": title,
                    "category": category_sel,
                    "main_category": main_category_sel,
                    "currency": currency_code,
                    "goal": float(goal_amount),
                    "country": country_code,
                    "usd_goal_real": float(usd_goal_real),
                    "clean_text": clean_title,
                    "goal_log": float(goal_log),
                    "usd_goal_real_log": float(usd_goal_real_log),
                    "title_length": int(title_length),
                    "title_word_count": int(title_word_count),
                    "campaign_duration": int(campaign_duration),
                    "launch_year": int(launch_year),
                    "launch_month": int(launch_month),
                    "launch_day": int(launch_day),
                    "launch_weekday": int(launch_weekday),
                    "launch_quarter": int(launch_quarter),
                    "goal_per_day": float(goal_per_day),
                }])
                
                # Double-check sorting order
                input_df = input_df[list(tabular.feature_names_in_)]
                
                # 5. Transform features using saved transformers
                X_text = vectorizer.transform(input_df["clean_text"])
                X_tabular = tabular.transform(input_df)
                
                # 6. Fuse using scipy hstack
                X_fused = hstack([X_text, X_tabular], format="csr")
                
                # 7. Predict success and probability
                pred_label = model.predict(X_fused)[0]
                
                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(X_fused)[0][1]
                else:
                    decision_score = model.decision_function(X_fused)[0]
                    probability = 1 / (1 + np.exp(-decision_score))
                    
                # 8. Compute confidence and status string
                is_success = pred_label == 1
                confidence_val = probability if is_success else (1.0 - probability)
                
                if confidence_val >= 0.80:
                    confidence_label = "Very High"
                    confidence_color = "green"
                elif confidence_val >= 0.65:
                    confidence_label = "High"
                    confidence_color = "green"
                elif confidence_val >= 0.50:
                    confidence_label = "Medium"
                    confidence_color = "orange"
                else:
                    confidence_label = "Low"
                    confidence_color = "red"
                    
            # ---------------------------------------------
            # PREDICTION RESULTS RENDER
            # ---------------------------------------------
            st.markdown("### 📊 Prediction Result")
            
            # Big Result Card
            if is_success:
                callout_class = "result-success"
                status_icon = "🚀"
                status_title = "Campaign Likely to Succeed"
                status_desc = f"Our model estimates an **{probability*100:.1f}%** chance of meeting or exceeding the goal."
            else:
                callout_class = "result-failure"
                status_icon = "❌"
                status_title = "Campaign Likely to Fail"
                status_desc = f"Our model estimates only an **{probability*100:.1f}%** chance of meeting the goal."
                
            st.markdown(
                f"""
                <div class="result-callout {callout_class}">
                    <div style="font-size: 2.5rem; line-height: 1;">{status_icon}</div>
                    <div>
                        <div style="font-weight: 700; font-size: 1.3rem;">{status_title}</div>
                        <div style="opacity: 0.9; margin-top: 4px;">{status_desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Sub layout for metrics
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                st.markdown("<h3>Probability Gauge</h3>", unsafe_allow_html=True)
                st.progress(float(probability))
                
                gauge_cols = st.columns(2)
                gauge_cols[0].metric("Success Probability", f"{probability*100:.2f}%")
                gauge_cols[1].metric("Decision Status", "SUCCESS" if is_success else "FAILURE")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with res_col2:
                st.markdown('<div class="saas-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("<h3>Confidence Level</h3>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="font-size: 2.2rem; font-weight: 800; color: {confidence_color}; margin-top: 10px;">
                        {confidence_label}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.caption(f"Model has **{confidence_val*100:.1f}%** confidence in this specific outcome based on fused features.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Summary Section
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("<h3>Campaign Summary</h3>", unsafe_allow_html=True)
            
            launch_weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][launch_weekday]
            launch_month_name = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][launch_month - 1]
            
            summary_html = f"""
            The project <b>"{title}"</b> is configured as a campaign in the <b>{category_sel}</b> subcategory under the <b>{main_category_sel}</b> group. 
            It targets a goal of <b>{goal_amount:,.2f} {currency_code}</b> (which is equivalent to <b>{usd_goal_real:,.2f} USD</b>). 
            The campaign is scheduled to run for <b>{campaign_duration} days</b>, launching on <b>{launch_weekday_name}, {launch_month_name} {launch_day}, {launch_year}</b> (Quarter {launch_quarter}).
            """
            st.write(summary_html)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Local Feature Importance
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("<h3>Local Explanation (Feature Contributions)</h3>", unsafe_allow_html=True)
            st.caption("This chart displays the specific features of your input that had the largest impact on this prediction. Positive coefficients push the prediction towards Success (Blue), while negative coefficients pull it towards Failure (Red).")
            
            # Calculate contributions
            text_features = list(vectorizer.get_feature_names_out())
            raw_tabular_features = tabular.get_feature_names_out()
            tabular_features = [
                f.replace("numeric__", "").replace("categorical__", "")
                for f in raw_tabular_features
            ]
            feature_names = text_features + tabular_features
            
            X_dense = X_fused.toarray()[0]
            local_contribs = X_dense * model.coef_[0]
            
            local_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': X_dense,
                'Coefficient': model.coef_[0],
                'Contribution': local_contribs
            })
            # Remove zero contributions
            local_df = local_df[local_df['Contribution'].abs() > 0.0001].copy()
            local_df['Importance'] = local_df['Contribution'].abs()
            local_df = local_df.sort_values(by='Importance', ascending=False)
            
            if local_df.empty:
                st.info("No significant features identified for this prediction.")
            else:
                col_chart, col_table = st.columns([3, 2])
                with col_chart:
                    # Plot top 10 local contributions
                    top_local = local_df.head(10).sort_values(by='Contribution')
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    fig.patch.set_facecolor('#ffffff')
                    ax.set_facecolor('#ffffff')
                    
                    colors = ['#ff3b30' if c < 0 else '#0071e3' for c in top_local['Contribution']]
                    bars = ax.barh(top_local['Feature'], top_local['Contribution'], color=colors, height=0.5)
                    
                    # Stylings
                    for spine in ['top', 'right', 'bottom', 'left']:
                        ax.spines[spine].set_visible(False)
                    ax.axvline(0, color='#86868b', linewidth=0.8, linestyle='--')
                    ax.tick_params(colors='#1d1d1f', labelsize=10)
                    ax.set_xlabel('Log-Odds Contribution', color='#1d1d1f', fontsize=10, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                with col_table:
                    st.markdown("##### Feature Contribution Table")
                    display_local = local_df.head(10)[['Feature', 'Contribution']].copy()
                    display_local['Direction'] = display_local['Contribution'].apply(lambda x: "🟢 Success" if x > 0 else "🔴 Failure")
                    display_local['Contribution'] = display_local['Contribution'].map("{:+.4f}".format)
                    st.dataframe(display_local, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# PAGE 3: PERFORMANCE DASHBOARD
# ==========================================================

elif page == "Dashboard":
    st.markdown("### 📈 Model Evaluation & Analytics")
    
    # Tabs for different dashboard views
    dash_tabs = st.tabs(["Performance Metrics", "Evaluation Plots", "Global Feature Importance"])
    
    with dash_tabs[0]:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h3>Model Performance Summary</h3>", unsafe_allow_html=True)
        st.caption("Trained on Kickstarter raw datasets using logistic regression classifier tuned with 5-fold CV.")
        
        metrics_summary = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            "Value": ["68.57%", "59.45%", "69.76%", "64.19%", "75.61%"],
            "Description": [
                "Percentage of overall correct predictions on the test set.",
                "Out of all predicted successful campaigns, how many actually succeeded.",
                "Out of all actual successful campaigns, how many were correctly predicted.",
                "Harmonic mean of Precision and Recall, balancing both false positives and false negatives.",
                "Area Under the ROC Curve, measuring how well the model separates the two classes."
            ]
        })
        st.table(metrics_summary)
        
        st.markdown("##### Detailed Classification Report")
        report_path = os.path.join(project_root, "reports/tables/classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_txt = f.read()
            st.code(report_txt)
        else:
            st.info("Classification report file not found.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with dash_tabs[1]:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h3>Pre-computed Evaluation Plots</h3>", unsafe_allow_html=True)
        st.caption("Figures generated from the validation split during training pipeline run.")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        
        cf_path = os.path.join(project_root, "reports/figures/confusion_matrix.png")
        roc_path = os.path.join(project_root, "reports/figures/roc_curve.png")
        pr_path = os.path.join(project_root, "reports/figures/precision_recall_curve.png")
        
        with col_p1:
            if os.path.exists(cf_path):
                st.image(cf_path, caption="Confusion Matrix", use_container_width=True)
            else:
                st.warning("Confusion Matrix plot not found.")
                
        with col_p2:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
            else:
                st.warning("ROC Curve plot not found.")
                
        with col_p3:
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Precision-Recall Curve", use_container_width=True)
            else:
                st.warning("Precision-Recall Curve plot not found.")
                
        st.markdown('</div>', unsafe_allow_html=True)
        
    with dash_tabs[2]:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h3>Global Feature Coefficients</h3>", unsafe_allow_html=True)
        st.caption("Global feature importance coefficients derived from the Logistic Regression model parameters.")
        
        if global_importance_df is not None:
            col_g_chart, col_g_table = st.columns([3, 2])
            
            with col_g_chart:
                # Plot top positive and negative features
                top_pos = global_importance_df[global_importance_df['Coefficient'] > 0].head(10)
                top_neg = global_importance_df[global_importance_df['Coefficient'] < 0].head(10)
                top_features = pd.concat([top_pos, top_neg]).sort_values(by='Coefficient')
                
                fig, ax = plt.subplots(figsize=(9, 6))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                colors = ['#ff3b30' if c < 0 else '#0071e3' for c in top_features['Coefficient']]
                bars = ax.barh(top_features['Feature'], top_features['Coefficient'], color=colors, height=0.55)
                
                for spine in ['top', 'right', 'bottom', 'left']:
                    ax.spines[spine].set_visible(False)
                ax.axvline(0, color='#86868b', linewidth=0.8, linestyle='--')
                ax.tick_params(colors='#1d1d1f', labelsize=9)
                ax.set_xlabel('Global Model Coefficient', color='#1d1d1f', fontsize=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            with col_g_table:
                st.markdown("##### Feature Importance Table")
                st.dataframe(
                    global_importance_df.head(25)[['Feature', 'Type', 'Coefficient']],
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("Global feature importance CSV table not found.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# PAGE 4: ABOUT PAGE
# ==========================================================

else:
    st.title("📖 Project Overview & Documentation")
    st.caption("Technical walkthrough of the training pipeline, feature engineering rules, and prediction systems.")
    
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("<h3>Dataset Overview</h3>", unsafe_allow_html=True)
    st.write(
        """
        The model is trained on the classic **Kickstarter Projects dataset** from Kaggle, which details campaign attributes and ultimate outcomes (success vs failure).
        Campaign states other than <code>successful</code> and <code>failed</code> (e.g. canceled, suspended, live) are filtered out, treating the problem as a clean **binary classification** task.
        """, 
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("<h3>End-to-End Pipeline Architecture</h3>", unsafe_allow_html=True)
    
    # Design diagram in text
    st.code(
        """
        [Raw Inputs] ---> Text Cleaning (TextCleaner) ---> TF-IDF (5,000 Vocab) ---\n"
        "                                                                           |---> Feature Fusion (hstack) ---> Logistic Regression Classifier\n"
        "[Raw Inputs] ---> Feature Engineering -------> Tabular Preprocessor ------/
        """,
        language="text"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("<h3>Feature Engineering Rules</h3>", unsafe_allow_html=True)
    st.write("The following 11 features are engineered from the raw input inputs and processed through the ColumnTransformer:")
    
    fe_data = pd.DataFrame({
        "Feature Name": [
            "goal_log",
            "usd_goal_real_log",
            "title_length",
            "title_word_count",
            "campaign_duration",
            "launch_year",
            "launch_month",
            "launch_day",
            "launch_weekday",
            "launch_quarter",
            "goal_per_day"
        ],
        "Formula / Extraction Rules": [
            "log1p(goal)",
            "log1p(usd_goal_real) where usd_goal_real is goal converted using currency FX rates",
            "Character length of the original raw campaign title",
            "Word count of the original raw campaign title (split on whitespace)",
            "Difference in days between deadline date and launched date",
            "Calendar year when the campaign launched",
            "Calendar month when the campaign launched (1 to 12)",
            "Day of the month when the campaign launched (1 to 31)",
            "Day of the week when the campaign launched (0 = Monday, 6 = Sunday)",
            "Calendar quarter when the campaign launched (1 to 4)",
            "goal_log divided by (campaign_duration + 1)"
        ]
    })
    st.table(fe_data)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("<h3>Modeling & Validation Details</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        - **Text Vectorization**: A `TfidfVectorizer` transforms the cleaned titles using **unigrams and bigrams**, restricted to the top **5,000 features** with sublinear term-frequency scaling.
        - **Tabular Preprocessing**: Uses a `ColumnTransformer` to impute missing values (median for numeric, mode for categorical), scale numeric inputs (`StandardScaler`), and one-hot encode categories (`OneHotEncoder` ignoring unknown categories during inference).
        - **Classifier**: Logistic Regression solved using `liblinear` and optimized via grid search (`GridSearchCV`) targeting the F1 score.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)