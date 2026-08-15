"""
Kickstarter Campaign Intelligence System - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

# Ensure root directory is in PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.consultant.report_generator import ReportGenerator

# Page config
st.set_page_config(
    page_title="Kickstarter Campaign Intelligence",
    page_icon="🚀",
    layout="wide",
)

# Load pipeline models and categories
@st.cache_resource
def load_system_pipeline():
    generator = ReportGenerator()
    # Extract supported categories from the fitted preprocessor
    preprocessor = generator.predictor.preprocessor.preprocessor
    encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
    categories = encoder.categories_
    
    countries = sorted(list(categories[0]))
    currencies = sorted(list(categories[1]))
    subcategories = sorted(list(categories[2]))
    main_categories = sorted(list(categories[3]))
    
    return generator, countries, currencies, subcategories, main_categories

# Embed custom CSS for modern aesthetics
st.markdown(
    """
    <style>
    .metric-container {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #334155;
    }
    .custom-card {
        background-color: #0f172a;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #1e293b;
        margin-bottom: 15px;
    }
    .custom-title {
        font-family: 'Google Fonts', sans-serif;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 2px;
    }
    .success-alert {
        padding: 10px;
        background-color: #022c22;
        color: #34d399;
        border-radius: 6px;
        border-left: 5px solid #10b981;
    }
    .failed-alert {
        padding: 10px;
        background-color: #450a0a;
        color: #f87171;
        border-radius: 6px;
        border-left: 5px solid #ef4444;
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    generator, countries, currencies, subcategories, main_categories = load_system_pipeline()
except Exception as e:
    st.error(f"Error loading model dependencies: {e}")
    st.stop()

# Header Section
st.markdown("<h1 class='custom-title'>🚀 Kickstarter Campaign Intelligence System</h1>", unsafe_allow_html=True)
st.caption("End-to-End Explainable AI decision-support system | Semantic RAG + XGBoost + SHAP + Local LLM Consultant")

st.divider()

# Sidebar inputs
st.sidebar.header("Campaign Launch Parameters")

title = st.sidebar.text_area(
    "Campaign Pitch / Tagline",
    value="AI-powered fitness application with personalized workouts, real-time workout tracking, and AI coaching",
    placeholder="Describe the campaign concept...",
    help="This description is embedded using Sentence Transformers to retrieve historically similar campaigns."
)

col_cat1, col_cat2 = st.sidebar.columns(2)
with col_cat1:
    main_cat = st.selectbox(
        "Main Category",
        options=main_categories,
        index=main_categories.index("Technology") if "Technology" in main_categories else 0
    )
with col_cat2:
    sub_cat = st.selectbox(
        "Sub Category",
        options=subcategories,
        index=subcategories.index("Apps") if "Apps" in subcategories else 0
    )

col_geo1, col_geo2 = st.sidebar.columns(2)
with col_geo1:
    country = st.selectbox(
        "Target Country",
        options=countries,
        index=countries.index("US") if "US" in countries else 0
    )
with col_geo2:
    currency = st.selectbox(
        "Currency",
        options=currencies,
        index=currencies.index("USD") if "USD" in currencies else 0
    )

goal = st.sidebar.number_input(
    "Funding Goal",
    min_value=100.0,
    max_value=10000000.0,
    value=15000.0,
    step=500.0,
    format="%.2f"
)

duration = st.sidebar.slider(
    "Campaign Duration (Days)",
    min_value=1,
    max_value=90,
    value=30
)

# Analyze Button
run_analysis = st.sidebar.button("Run Full Campaign Analysis", use_container_width=True)

if run_analysis or "facts" in st.session_state:
    if not title.strip():
        st.sidebar.warning("Please enter a campaign pitch.")
        st.stop()
        
    if run_analysis:
        # Run pipeline step-by-step with status logs
        with st.status("Analyzing campaign parameters...", expanded=True) as status:
            st.write("🔍 Retrieving semantically similar campaigns from ChromaDB...")
            retrieval_results = generator.retriever.search(title, top_k=50)
            
            st.write("📈 Running recommendation engine & risk analysis...")
            recommendation_results = generator.recommender.recommend(title, top_k=50)
            funding_results = generator.optimizer.optimize(title, top_k=50)
            risk_results = generator.risk_analyzer.analyze(title)
            
            st.write("🤖 Fusing text embeddings with engineered tabular launch features...")
            X_sample = generator.predictor.get_features(
                title=title,
                category=sub_cat,
                main_category=main_cat,
                country=country,
                currency=currency,
                goal=goal,
                duration=duration,
            )
            
            st.write("🔮 Predicting success probability using XGBoost (93.56% accurate classifier)...")
            prediction_results = generator.predictor.predict(
                title=title,
                category=sub_cat,
                main_category=main_cat,
                country=country,
                currency=currency,
                goal=goal,
                duration=duration,
            )
            
            st.write("🧬 Computing feature contributions via SHAP TreeExplainer...")
            explanation_results = generator.explanation_engine.explain_instance(X_sample)
            
            st.write("💬 Generating grounded strategic verdict with local Llama 3.2...")
            # Compile facts dictionary
            raw_facts = {
                "campaign": {
                    "title": title,
                    "category": sub_cat,
                    "main_category": main_cat,
                    "country": country,
                    "currency": currency,
                    "funding_goal": goal,
                    "duration": duration
                },
                "prediction": prediction_results,
                "risk": risk_results,
                "funding": funding_results,
                "similar_campaigns": [
                    {
                        "title": c["title"],
                        "similarity": c["similarity"],
                        "category": c["category"],
                        "main_category": c["main_category"],
                        "goal": c["goal"],
                        "pledged": c["pledged"],
                        "state": c["state"],
                        "country": c.get("country", "")
                    }
                    for c in retrieval_results[:5]
                ],
                "shap": {
                    "positive_factors": explanation_results.get("positive_factors", [])[:5],
                    "negative_factors": explanation_results.get("negative_factors", [])[:5]
                },
                "business_recommendations": recommendation_results.get("recommendations", [])
            }
            facts = generator._to_python(raw_facts)
            prompt = generator.prompt_builder.build_prompt(facts)
            
            verdict = generator.consultant.ask(prompt)
            
            # Validate output and self-correct if required
            validation_errors = generator._validate_verdict(verdict, facts)
            retry_occurred = False
            if validation_errors:
                retry_occurred = True
                st.write("⚠️ Contradiction detected in LLM output. Triggering self-correction retry...")
                retry_prompt = prompt + (
                    f"\n\n[SYSTEM NOTE: Your previous response contained the following factual contradictions with the data:\n"
                    + "\n".join([f"- {err}" for err in validation_errors])
                    + "\n\nPlease rewrite the response to correct these contradictions. Make sure every success percentage, risk score/level, and recommended funding goal matches the JSON data exactly.]"
                )
                verdict = generator.consultant.ask(retry_prompt)
                validation_errors = generator._validate_verdict(verdict, facts)
                
            status.update(label="Campaign Analysis Complete!", state="complete", expanded=False)
            
        # Store in session state to persist on tab switching
        st.session_state["facts"] = facts
        st.session_state["prediction"] = prediction_results
        st.session_state["risk"] = risk_results
        st.session_state["funding"] = funding_results
        st.session_state["retrieval"] = retrieval_results
        st.session_state["shap"] = explanation_results
        st.session_state["recommendation"] = recommendation_results
        st.session_state["verdict"] = verdict
        st.session_state["validation_errors"] = validation_errors
        st.session_state["retry_occurred"] = retry_occurred

    # Retrieve values from session state
    facts = st.session_state["facts"]
    prediction = st.session_state["prediction"]
    risk = st.session_state["risk"]
    funding = st.session_state["funding"]
    retrieval = st.session_state["retrieval"]
    shap = st.session_state["shap"]
    recommendation = st.session_state["recommendation"]
    verdict = st.session_state["verdict"]
    validation_errors = st.session_state["validation_errors"]
    retry_occurred = st.session_state["retry_occurred"]

    # Display clean tabs
    tab_pred, tab_rag, tab_bi, tab_consultant = st.tabs([
        "🔮 Success Prediction & SHAP",
        "🔍 Similar Campaigns (RAG)",
        "💰 Funding & Risk Optimization",
        "🤖 AI Consultant Verdict"
    ])

    # Tab 1: Predictions & SHAP
    with tab_pred:
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            st.subheader("Model Prediction Output")
            pred_outcome = prediction["prediction"]
            success_prob = prediction["success_probability"]
            
            if pred_outcome == "Successful":
                st.markdown(
                    f"<div class='success-alert'><strong>PREDICTED OUTCOME: SUCCESSFUL</strong><br>The classifier predicts a success probability of {success_prob}% based on campaign parameters.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='failed-alert'><strong>PREDICTED OUTCOME: FAILED</strong><br>The classifier predicts a failure probability of {100 - success_prob:.2f}% (Success Probability: {success_prob}%).</div>",
                    unsafe_allow_html=True
                )
                
            st.write("")
            st.metric("Success Probability", f"{success_prob}%", help="Calculated using the XGBoost fused features model.")
            st.progress(success_prob / 100.0)
            
            st.write("")
            st.subheader("Risk Score Assessment")
            risk_level = risk["risk_level"]
            risk_score = risk["risk_score"]
            
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Risk Level", risk_level)
            col_r2.metric("Risk Score", f"{risk_score}/100")
            
            st.write("**Key Indicators:**")
            for reason in risk.get("reasons", []):
                st.write(f"- ⚠️ {reason}")
                
        with col_pred2:
            st.subheader("SHAP Feature Influence (XAI)")
            st.caption("Horizontal bars indicate contribution to success (green) or failure (red) on a log-odds scale.")
            
            pos_factors = shap.get("positive_factors", [])[:5]
            neg_factors = shap.get("negative_factors", [])[:5]
            
            # Build horizontal bar chart using Matplotlib
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#0f172a')  # Dark theme background
            ax.set_facecolor('#0f172a')
            
            features = []
            impacts = []
            colors = []
            
            # Negatives first (drivers towards failure)
            for f in reversed(neg_factors):
                features.append(f["feature"])
                impacts.append(f["shap_value"])
                colors.append('#ef4444')  # Red
                
            # Positives
            for f in pos_factors:
                features.append(f["feature"])
                impacts.append(f["shap_value"])
                colors.append('#10b981')  # Green
                
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, impacts, color=colors, height=0.6)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, color='#f1f5f9', fontsize=9)
            ax.axvline(0, color='#64748b', linewidth=1, linestyle='--')
            
            ax.tick_params(colors='#f1f5f9', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.set_xlabel("SHAP Value (Impact Magnitude)", color='#f1f5f9', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)

    # Tab 2: Similar Campaigns
    with tab_rag:
        st.subheader("Semantic Search Summaries")
        
        c_r1, c_r2, c_r3, c_r4 = st.columns(4)
        c_r1.metric("Retrieved References", len(retrieval))
        
        successful_refs = sum(1 for c in retrieval if c["state"] == "successful")
        c_r2.metric("Successful References", successful_refs)
        c_r3.metric("Failed References", len(retrieval) - successful_refs)
        
        rate = round(successful_refs / len(retrieval) * 100, 1) if retrieval else 0.0
        c_r4.metric("Historic Reference Success Rate", f"{rate}%")
        
        st.divider()
        st.subheader("Top 10 Conceptually Similar Campaigns")
        st.caption("Identified using 384-dimensional dense vectors and cosine distance.")
        
        similar_df = pd.DataFrame([
            {
                "Campaign Title": c["title"],
                "Similarity %": f"{c['similarity']*100:.1f}%",
                "Category": c["category"],
                "Main Category": c["main_category"],
                "Goal": f"{c.get('country', '')} {c['goal']:,.0f}",
                "Pledged": f"{c.get('country', '')} {c['pledged']:,.0f}",
                "State": "Successful" if c["state"] == "successful" else "Failed"
            }
            for c in retrieval[:10]
        ])
        st.dataframe(similar_df, use_container_width=True)

    # Tab 3: Funding & Risk
    with tab_bi:
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.subheader("Funding Goal Optimizer")
            st.write("We compare your goal against successful references to recommend a target with the highest statistical likelihood of funding:")
            
            st.metric("Recommended Goal", f"USD {funding.get('recommended_goal', 0):,.2f}", help="The median goal of successful conceptually similar campaigns.")
            
            st.write(f"- **Proposed Goal**: {currency} {goal:,.2f}")
            st.write(f"- **Successful Campaign Average Goal**: USD {funding.get('average_goal', 0):,.2f}")
            st.write(f"- **Successful Goal Range**: USD {funding.get('minimum_goal', 0):,.2f} to USD {funding.get('maximum_goal', 0):,.2f}")
            st.write(f"- **Successful Campagins Compared**: {funding.get('successful_campaigns', 0)} out of {funding.get('total_compared', 0)}")
            
        with col_f2:
            st.subheader("Actionable Recommendations")
            for rec in recommendation.get("recommendations", []):
                st.write(f"- 💡 {rec}")
                
            st.write("")
            st.subheader("Risk Analyzer Suggestions")
            for sugg in risk.get("suggestions", []):
                st.write(f"- ✅ {sugg}")

    # Tab 4: AI Consultant
    with tab_consultant:
        st.subheader("AI Strategy Consultant Verdict")
        
        if retry_occurred:
            st.info("ℹ️ **Validation Check Status**: LLM generated output failed initial validation checks. The correction engine successfully retried Ollama and verified the final report.")
            
        if validation_errors:
            st.warning(f"⚠️ **Factual Warning**: Discrepancies detected against python calculations: {validation_errors}")
            
        st.markdown(verdict)
        
        # Add report download option
        report_text = f"""==================================================
KICKSTARTER CAMPAIGN INTELLIGENCE REPORT
==================================================
Deterministic calculations:
- Title: {title}
- Success Probability: {prediction['success_probability']}%
- Recommended Goal: USD {funding.get('recommended_goal', 0):,.2f}
- Risk Level: {risk['risk_level']} (Score: {risk['risk_score']}/100)

AI CONSULTANT VERDICT:
{verdict}
"""
        st.download_button(
            label="Download Strategic Report",
            data=report_text,
            file_name="kickstarter_strategy_report.txt",
            mime="text/plain",
            use_container_width=True
        )
else:
    # App is in initial state
    st.info("👈 Set campaign launch parameters in the sidebar and click **Run Full Campaign Analysis** to trigger the pipeline.")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.subheader("1. Semantic RAG Search")
        st.write("Sentence Transformers embed your campaign tagline and ChromaDB retrieves conceptually matching historic references.")
        
    with col_e2:
        st.subheader("2. XGBoost & SHAP")
        st.write("Fuses your description embedding with 225 tabular variables. XGBoost predicts success probability, and SHAP explains model decisions.")
        
    with col_e3:
        st.subheader("3. Grounded local LLM")
        st.write("Llama 3.2 acts as a strategic consultant, generating advice grounded strictly in Python's calculations, validated by a retry layer.")