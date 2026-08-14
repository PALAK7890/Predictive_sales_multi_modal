# Kickstarter Campaign Intelligence System
An end-to-end Explainable AI decision-support system combining semantic retrieval, RAG, feature fusion, XGBoost, SHAP, and a locally hosted LLM consultant.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDatabase-ChromaDB-blueviolet.svg)](https://www.trychroma.com/)
[![Transformers](https://img.shields.io/badge/Embeddings-Sentence--Transformers-yellow.svg)](https://sbert.net/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Ollama](https://img.shields.io/badge/LLM%20Inference-Ollama-lightgrey.svg)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)

---

## What is this?
Most campaign prediction systems stop at predicting success or failure. This system combines historical campaign retrieval, machine learning, explainability, funding optimization, risk analysis, and grounded LLM reasoning into a single decision-support pipeline. 

### Why does it exist?
Crowdfunding campaigns represent a high-stakes decision for creators. Before launching, they need to know if their funding goal is realistic, how similar campaigns performed, what risks their project setup carries, and *why* their project is predicted to succeed or fail.

### What makes it technically interesting?
Instead of a simple classifier or a chatty LLM wrapper, this system integrates a deterministic machine learning and explainability pipeline directly with a locally hosted large language model (Ollama + Llama 3.2). Python remains the absolute source of truth for numerical calculations, while the LLM acts strictly as a narrative interpreter. A lightweight regex validation layer monitors LLM output, triggering automated retries if any numerical contradictions are generated.

---

## Why This Project is Different

Traditional crowdfunding analysis tools typically rely on a single-stage model:
```text
Campaign Idea ──> ML Classifier ──> Binary Prediction (Success/Failure)
```
This project implements an end-to-end multi-stage pipeline where historical evidence, machine learning, and explainability feed into each other to provide a cohesive strategy:

```text
Campaign Idea
      │
      ▼
Semantic Retrieval (RAG) ──> Finds top 50 historically similar campaigns
      │
      ├───────────────────────> Business Recommendation Engine
      ├───────────────────────> Funding Goal Optimizer (median-based)
      └───────────────────────> Risk Analyzer (rules-based scoring)
      │
      ▼
Embedding & Tabular Fusion ──> Stacks text embeddings (384-d) & tabular features (225-d)
      │
      ▼
XGBoost Classifier ──> Generates success & failure probabilities
      │
      ▼
SHAP TreeExplainer ──> Calculates feature impact on prediction (mapped to clean names)
      │
      ▼
Structured Facts Object ──> Compiles all calculations into a deterministic JSON schema
      │
      ▼
Grounded LLM Consultant ──> Interprets facts qualitatively via local Ollama
      │
      ▼
Validation & Retry Layer ──> Audits output for contradictions; retries Ollama if mismatched
      │
      ▼
Final Business Report ──> Outputs authoritative data + verified strategic advice
```

This multi-stage architecture ensures that predictions are always explainable, business metrics are backed by historical reference campaigns, and the AI consultant's advice is strictly grounded in deterministic statistics.

---

## Layered System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Kickstarter Dataset] --> B[Data Cleaning & Feature Engineering]
    end

    subgraph Retrieval Layer (RAG)
        B --> C[Sentence Transformer all-MiniLM-L6-v2]
        C --> D[ChromaDB Vector Store]
        D --> E[Semantic Search & Lookup]
    end

    subgraph Intelligence Layer
        E --> F[Recommendation Engine]
        E --> G[Funding Goal Optimizer]
        E --> H[Risk Analyzer]
    end

    subgraph ML Layer
        B --> I[Tabular Feature Extraction]
        C --> I[Embedding & Tabular Feature Fusion]
        I --> J[XGBoost Classifier]
    end

    subgraph Explainability Layer (XAI)
        J --> K[SHAP TreeExplainer & Feature Mapper]
    end

    subgraph LLM Layer
        F --> L[Structured Facts Object]
        G --> L
        H --> L
        J --> L
        K --> L
        L --> M[Prompt Builder]
        M --> N[Ollama Llama 3.2]
        N --> O[Grounding Validation & Retry]
    end

    subgraph Output
        O --> P[Final Campaign Intelligence Report]
    end
```

---

## Technical Highlights

### Semantic Retrieval
Instead of keyword matching, which misses conceptual similarities, the retrieval layer embeds campaign descriptions into a 384-dimensional dense vector space using `all-MiniLM-L6-v2`. These embeddings are queried against a persistent local ChromaDB collection using **cosine similarity** to locate the 50 conceptually closest campaign references.

### Feature Fusion
To predict campaign success, text descriptions must be evaluated alongside structured project data. The feature fusion pipeline combines:
* **Semantic Embeddings**: 384-dimensional dense vectors representing the campaign text.
* **Tabular Features**: 225-dimensional sparse representations covering category, main category, country, currency, launch timeline, title length, and goal magnitudes (imputed and standard-scaled).
* **Fusable Dimensions**: Yields a final fused matrix of **609 features** for all **331,675 campaigns** in the dataset.

### XGBoost Success Predictor
For structured, fused datasets, tree boosting outperforms deep learning. We use an `XGBClassifier` trained on the 609-feature fused matrix to predict success. By adjusting parameters (learning rate `0.05`, estimators `300`, subsample `0.8`), the classifier achieves high generalization on out-of-fold evaluations.

### SHAP Explainability
Deep learning and ensemble trees are traditionally "black boxes." We use a SHAP `TreeExplainer` to calculate Shapley values for the campaign's features, isolating why the model predicted success or failure. A mapping class translates sparse, encoded indices (e.g. `numeric__backers` or `categorical__country_US`) into clean labels (e.g. `"Number of Backers"`, `"Country: US"`) and splits them into positive (success-favoring) and negative (failure-favoring) drivers.

### Grounded LLM Consultant (Local Inference)
The local Ollama Llama 3.2 model does not calculate or guess campaign numbers. It receives a serialized JSON facts object. The prompt restricts the LLM to narrative interpretation, meaning it behaves strictly as a strategy consultant that translates the numerical facts (probabilities, goals, SHAP factors) into actionable pre-launch advice.

### Validation Retry Layer
LLMs are prone to hallucinating numbers under formatting pressure. To prevent this, the report generator runs a regex-based validation check on the LLM's response. If Llama 3.2 writes a recommended goal, success rate, or risk score that contradicts the ML pipeline:
1. It flags the validation errors.
2. It appends the correction instruction back to the model.
3. It retries Ollama once.
4. If it fails a second time, it prefixes the final verdict with a warning block showing the detected contradictions, ensuring Python remains the authoritative source.

---

## Project Scale Metrics

* **Dataset Size**: 331,675 campaigns
* **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
* **Preprocessed Features**: 225-dimensional engineered tabular metrics
* **Fused Input Dimension**: 609 features
* **Success Success Rate (Base)**: ~35% successful campaigns
* **XGBoost Accuracy**: **93.56%**

---

## Machine Learning Results

The XGBoost success classifier achieves the following performance on a stratified 20% test split:

| Metric | Result |
| :--- | :---: |
| **Accuracy** | 93.56% |
| **Class 0 (Failed) Precision** | 0.95 |
| **Class 0 (Failed) Recall** | 0.94 |
| **Class 0 (Failed) F1-Score** | 0.95 |
| **Class 1 (Successful) Precision** | 0.91 |
| **Class 1 (Successful) Recall** | 0.93 |
| **Class 1 (Successful) F1-Score** | 0.92 |

*Note: These results reflect the current dataset and evaluation setup and should not be interpreted as a guarantee of real-world campaign success.*

---

## Demo Walkthrough

### Example Campaign Input
* **Title**: `"AI-powered fitness application with personalized workouts, real-time workout tracking, and AI coaching"`
* **Category**: `"Apps"` (Main Category: `"Technology"`)
* **Goal**: `$15,000` | **Duration**: `30 days` | **Country**: `"US"`

### Deterministic Output Preview
* **Success Probability**: `0.04%` | **Failure Probability**: `99.96%`
* **Risk Score**: `60/100` (Level: `MEDIUM`)
* **Recommended Funding Goal**: `$15,000` (Median of successful similar campaigns)
* **Average Successful Campaign Goal**: `$29,711`
* **Top Negative SHAP Factor**: `Number of Backers` (Impact: `-3.2659`)
* **Top Positive SHAP Factor**: `Country: Unknown` (Impact: `+2.6121`)

### AI Consultant Verdict Summary
The local Llama 3.2 model interprets these findings into strategic advice:
1. **Low Baseline Probability (0.04%)**: Reflects that almost all apps with similar descriptions historically failed due to crowded spaces and high goals.
2. **Backer Volatility**: Explains that `Number of Backers` is the model's most sensitive feature. Recommends building an email list of at least 1,000 leads prior to launch to guarantee Day 1 backers.
3. **Funding Realism**: Suggests maintaining the $15,000 goal (as it matches the median of successful references) but notes that successful apps raised $29,711 on average, warning that a smaller goal might restrict features.

---

## Project Structure

```text
Summer_Project
├── app/                           # Streamlit UI Components
│   ├── components/                # Metric cards and layouts
│   ├── pages/                     # Similar campaign explorer page
│   └── app.py                     # App entry point
├── artifacts/
│   ├── models/                    # Pickled ML classifiers & baseline models
│   │   └── xgboost.pkl            # Final trained XGBoost success predictor
│   ├── preprocessors/
│   │   └── xgb_preprocessor.pkl   # Fitted tabular preprocessor transformer
│   └── reports/
│       ├── figures/               # Matplotlib EDA charts
│       └── tables/                # Descriptive statistics tables
├── configs/
│   ├── config.py                  # Pipeline configuration constants
│   └── paths.py                   # Global system path mappings
├── data/
│   ├── interim/                   # Preprocessed interim CSV datasets
│   └── raw/                       # Ignored raw CSV datasets
├── docs/                          # Project documentation and resume assets
│   ├── DEMO.md                    # 3-minute recruiter demo script
│   ├── TECHNICAL_CASE_STUDY.md    # Detailed ML design case study
│   ├── RESUME.md                  # Pre-written resume bullets
│   ├── PORTFOLIO_DESCRIPTION.md   # Portfolio copy and elevator pitch
│   └── INTERVIEW_PREP.md          # Technical interview preparation Q&A
├── src/                           # System Source Code
│   ├── consultant/                # LLM Consultant wrapper & validation retry
│   ├── explainability/            # SHAP calculations and index mapping
│   ├── models/                    # XGBoost success prediction
│   ├── preprocessing/             # Tabular engineering & clean-text prep
│   ├── recommendation/            # Analytics, risk analyzer, and goal optimizer
│   ├── retrieval/                 # Semantic campaign retrieval & vector search
│   └── utils/                     # System loggers and metrics helpers
├── .gitignore                 # Standard repository ignore configuration
├── requirements.txt           # Audited Python package versions
├── test_consultant.py         # End-to-end pipeline run script
└── main.py                    # Original training pipeline entry
```

---

## Installation & Setup

### 1. Configure the Python Virtual Environment
Requires Python `3.13`.
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Configure Local Ollama model
Install Ollama from [ollama.com](https://ollama.com) and pull Llama 3.2:
```bash
ollama pull llama3.2
```
Verify the Ollama local daemon is running on port `11434` before running the test script.

---

## Running the System

To run the end-to-end Campaign Intelligence pipeline, parse features, run SHAP, query Ollama, and generate the validated report:
```bash
python test_consultant.py
```
The resulting report is saved locally to `artifacts/reports/campaign_report.txt`.

---

## Screenshots

### Similar Campaign Dashboard
*(Place campaign query Streamlit explorer screenshots here)*
* [Placeholder for similar_campaign_dashboard.png](file:///Users/PalakNst/Desktop/Summer_Project/docs/images/similar_campaign_dashboard.png)

### Campaign Intelligence Report
*(Place generated report outputs here)*
* [Placeholder for campaign_intelligence_report.png](file:///Users/PalakNst/Desktop/Summer_Project/docs/images/campaign_intelligence_report.png)

### SHAP Explanation
*(Place SHAP force/summary plot visualization here)*
* [Placeholder for shap_explanation.png](file:///Users/PalakNst/Desktop/Summer_Project/docs/images/shap_explanation.png)

### Funding & Risk Analysis
*(Place risk analysis gauge or goal distribution plots here)*
* [Placeholder for funding_risk_analysis.png](file:///Users/PalakNst/Desktop/Summer_Project/docs/images/funding_risk_analysis.png)

*To capture these, run the Streamlit dashboard app locally and save screenshots in `docs/images/`.*

---

## Engineering Challenges

### 1. OpenMP / macOS Threading Conflict
**Issue**: Running PyTorch (inside Sentence Transformers) followed by XGBoost (unpickling the model via Joblib) on macOS caused a silent segmentation fault during initialization. Both libraries compiled against independent, conflicting OpenMP threading runtimes (`libomp` vs `libiomp5`).
**Solution**: Found that importing `xgboost` at the absolute top of the entrypoint file before `sentence_transformers` or `torch` allows the runtimes to initialize compatibly, avoiding the thread-level crash.

### 2. Tabular Feature Preprocessor Mismatch
**Issue**: The feature fusion pipeline expected specific engineered features (e.g. `goal_log`, `usd_goal_real_log`, `campaign_duration`, etc.) that were not present in the clean text CSV.
**Solution**: Refactored the preprocessing pipeline `xgb_preprocessor.py` to calculate these engineered variables dynamically during inference, matching the training features index-for-index.

### 3. Large File Asset Staging
**Issue**: The campaign embeddings (`campaign_embeddings.pkl`, 486MB), fused matrices (`X_embedding_fused.npz`, 562MB), and ChromaDB persistence directories (879MB) exceeded GitHub's 100MB file size limit.
**Solution**: Untracked the large files from the Git index (`git rm --cached`) and rewrote `.gitignore` to protect the repository while keeping local copies intact for testing, adding instructions on how to regenerate them.

### 4. LLM Hallucinations on Numeric Outputs
**Issue**: Llama 3.2 occasionally confused numerical metrics (e.g. recommending a goal of `$29,711` which was the average, instead of the median recommendation of `$15,000`).
**Solution**: Created a structured JSON facts object to decouple numbers from LLM reasoning, made Python-generated reports authoritative, and built a validation-retry loop in `report_generator.py` to audit and correct the LLM outputs dynamically.

---

## Limitations

* **Historical Bias**: Relies on Kickstarter 2018 logs; does not reflect modern crowdfunding trends.
* **Correlation vs Causality**: SHAP features represent statistical model importances, not direct physical causes of campaign success.
* **Local Inference Latency**: Local execution of Llama 3.2 is CPU/GPU dependent and may take up to 20-30 seconds depending on system specs.