# Technical Case Study - Kickstarter Campaign Intelligence System

An end-to-end Explainable AI (XAI) decision-support system designed to optimize Kickstarter pre-launch parameters using dense retrieval, ensemble-tree classification, feature fusion, and a validated local LLM consultant.

---

## 1. Problem Statement
Launch planning is a major risk for crowdfunding creators. Creators must decide:
* **Viability**: Is the core campaign description conceptually aligned with historically successful projects?
* **Financial Benchmarks**: What funding goal is realistic given comparable campaign outcomes?
* **Launch Parameters**: How do categorical features (category, main category, country, currency) influence success probabilities?
* **Strategic Adaptation**: How do creators turn black-box machine learning predictions into structured launch adjustments?

This case study reviews a multi-stage architecture that leverages RAG (Dense Retrieval), Tabular-Text Feature Fusion, XGBoost, SHAP, and a locally hosted Llama 3.2 model to provide grounded campaign intelligence.

---

## 2. Dataset
* **Source**: Cleaned Kickstarter campaign logs containing **331,675 projects**.
* **Attributes**: 
  * Identifiers: `id`, `name`
  * Categorical: `category`, `main_category`, `country`, `currency`
  * Temporal: `launched` (datetime), `deadline` (datetime)
  * Financial: `goal`, `usd_goal_real`, `pledged`, `usd_pledged_real`, `backers`
  * Target Class: `state` (mapped to binary labels: `successful` = 1, all others = 0)
  * Semantic: `clean_text` (preprocessed campaign name/tagline)

---

## 3. Data Processing & Tabular Preprocessing
We engineered a pipeline via `XGBPreprocessor` to transform raw inputs:
1. **Temporal Features**: Campaign duration calculated as `deadline - launched` (in days). Features like `launch_year`, `launch_month`, `launch_day`, `launch_weekday`, and `launch_quarter` are extracted.
2. **Title Length**: Character length (`title_length`) and word count (`title_word_count`) extracted from the project name.
3. **Log-Transforms**: Due to high skewness in goal targets, `goal_log` and `usd_goal_real_log` are generated using `np.log1p`.
4. **Funding Speed**: `goal_per_day` is calculated as `usd_goal_real / campaign_duration` to represent required funding velocity.
5. **Handling Categoricals**: Values for `country`, `currency`, `category`, and `main_category` are encoded using a fitted `OneHotEncoder(handle_unknown="ignore")`, producing **225 sparse binary features**.
6. **Scaling & Imputation**: Numerical variables are imputed using median values and scaled via `StandardScaler`.

---

## 4. Semantic Retrieval & Embedding Architecture
* **Model**: Sentence Transformers `all-MiniLM-L6-v2`. It encodes taglines into a **384-dimensional dense vector space**.
* **Vector Store**: Persisted ChromaDB collection `kickstarter_campaigns` using **Cosine Distance**.
* **RAG Pipeline**: 
  * The user's campaign concept is embedded using the Sentence Transformer.
  * ChromaDB is queried using the query vector.
  * Cosine distance is converted to similarity: $\text{Similarity} = 1 - \text{Distance}$.
  * The retriever looks up campaign IDs in the main cleaned CSV database to compile metadata.

---

## 5. Embedding + Tabular Feature Fusion
To classify campaign outcomes, text semantics must be combined with launch logistics (e.g. goal, category, timeline).
* **Fusion Stacking**: We stack the dense text embedding array with the preprocessed tabular feature matrix:
  $$\mathbf{X}_{\text{fused}} = \begin{bmatrix} \mathbf{X}_{\text{embeddings}} & \mathbf{X}_{\text{tabular}} \end{bmatrix}$$
* **Dimensions**: 384 semantic columns + 225 tabular columns = **609 fused features** for all 331,675 projects.

---

## 6. Machine Learning Success Classifier (XGBoost)
We use an ensemble of gradient-boosted decision trees, which natively handles sparse one-hot features and non-linear interactions better than deep architectures on tabular tasks.
* **Model Spec**: `XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")`.
* **Accuracy**: **93.56%** on a stratified 20% test split.
* **Precision/Recall (Out of Fold)**:
  * Class 0 (Failed): Precision `0.95`, Recall `0.94`, F-1 `0.95`
  * Class 1 (Successful): Precision `0.91`, Recall `0.93`, F-1 `0.92`

---

## 7. Explainability Layer (SHAP)
We apply Shapley values to decompose the model's prediction:
1. **TreeExplainer**: Runs a SHAP `TreeExplainer` on the trained XGBoost model.
2. **Index Mapping**: A clean-name dictionary maps index ranges. Since the embeddings occupy indices `0..383`, they are labeled as `"Campaign Description Semantics (Dim X)"`. Tabular features occupying indices `384..608` are mapped using `get_feature_names_out()` from the fitted preprocessor and cleaned of package prefixes (e.g., `categorical__category_Games` $\rightarrow$ `Category: Games`).
3. **Driver Splitting**: Drivers are sorted by magnitude and split into positive contributions ($\text{SHAP} > 0$, driving success) and negative contributions ($\text{SHAP} < 0$, driving failure).

---

## 8. Business Intelligence Metrics
Directly from the top 50 retrieved semantic neighbors, we calculate:
* **Goal Optimization**: Calculates the median goal of similar successful campaigns to establish a realistic target, avoiding outliers.
* **Risk Scoring**: Evaluates category clustering, historic failure rates, and neighbor frequencies to yield a risk score (0-100) and risk level (LOW, MEDIUM, HIGH).

---

## 9. Grounding and LLM Validation Architecture
To integrate local LLM reasoning safely, we built a Python-authoritative orchestration layer:

```text
                  Deterministic Python Calculations
          (Retrieve similar, predict probability, calculate SHAP)
                                │
                                ▼
                       Facts Dictionary (JSON)
                                │
                                ▼
                       Prompt Builder (System Rules)
                                │
                                ▼
                       Local Ollama (Llama 3.2)
                                │
                                ▼
                        LLM Raw Output
                                │
                                ▼
               Regex Validation Layer (Sentence-Based)
                                │
            ┌───────────────────┴───────────────────┐
            ▼ Pass                                  ▼ Fail
    Save Final Report                       Retry Ollama Once
                                                    │
                                                    ▼
                                            LLM Revised Output
                                                    │
                                            ┌───────┴───────┐
                                     Pass   ▼        Fail   ▼
                                      Save Report     Prepend Warning Block
```

### Validation Specifications:
1. **Sentence Splitting**: The LLM's response is split into sentences.
2. **Keyword Scan**: The validation class inspects any sentence containing metrics like `"recommended goal"`, `"success probability"`, `"risk score"`, or `"risk level"`.
3. **Contradiction Test**: Checks if any dollar values or percentages in those sentences conflict with the factual inputs (e.g. LLM recommending the user's initial goal of `$15,000` when the optimizer recommends `$29,711`).
4. **Retry Loop**: If a contradiction occurs, the system logs the issue, appends a system notification warning the LLM about the specific discrepancy, and requests a new generation.

---

## 10. Engineering Challenges
* **OpenMP / macOS Silent Crash**: Running Sentence Transformers (PyTorch) followed by XGBoost loading (via Joblib) caused a silent segmentation fault on macOS. This was caused by conflicting OpenMP runtimes (`libomp` vs `libiomp5`). We resolved this by explicitly placing `import xgboost` at the top of the entrypoint file to enforce compatible initialization.
* **Large Asset Handling**: The vector DB, embedding pickling, and fused feature matrices exceeded GitHub's 100MB file limit. We used `git rm --cached` to untrack them from Git index while preserving local caches, documenting how users can rebuild them.
* **LLM Hallucinations**: In early tests, Llama 3.2 confused funding metrics, recommending the mean goal when the median optimizer recommended a lower target. Decoupling the narrative verdict, making Python-generated sections authoritative, and adding a validation retry layer resolved this.

---

## 11. Limitations & Future Work
* **Limitations**: Bound to the Kickstarter 2018 dataset; does not account for modern crowdfunding behaviors. SHAP feature importances represent correlation, not causality.
* **Future Work**: Add support for dynamic web-scraping to refresh the dataset. Enhance risk predictions to factor in seasonal variations or category saturation.
