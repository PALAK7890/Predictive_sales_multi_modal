# Technical Interview Preparation Guide
## Kickstarter Campaign Intelligence System

This document contains high-probability technical interview questions and concise, interview-quality answers covering every technology and engineering decision made in this project.

---

## 1. Retrieval & RAG

### Q: Why did you choose Sentence Transformers instead of a simple TF-IDF or keyword search?
* **Answer**: TF-IDF and keyword searches rely on exact word matching, which fails to capture conceptual synonyms. Sentence Transformers (specifically `all-MiniLM-L6-v2`) encode the text into a dense 384-dimensional space, capturing semantic semantics. For example, a search for `"AI fitness app"` conceptually retrieves `"GymPad - A Fitness Tracking App"` and `"Anyone Can Train!!! The Fitness App Revolution"` even if they share zero overlapping keywords.

### Q: Why ChromaDB?
* **Answer**: ChromaDB is an open-source, lightweight, and embeddable vector database. It persists directly to the local filesystem without requiring separate cloud infrastructure (like Pinecone) or custom daemon setups, keeping the repository simple, reproducible, and easy to run locally while natively supporting metadata filtering and fast cosine similarity searches.

### Q: What does the similarity score represent?
* **Answer**: We use the Cosine Distance metric in ChromaDB. The cosine distance measures the angular distance between two normalized embeddings in the vector space, ranging from `0.0` (identical) to `2.0` (opposite). We convert this to Cosine Similarity ($\text{Similarity} = 1 - \text{Distance}$) to present a clean, intuitive percentage representation (typically `0.6` to `0.8` for relevant campaigns).

---

## 2. Feature Engineering & Fusion

### Q: Why combine dense text embeddings and tabular preprocessed features?
* **Answer**: Crowdfunding campaigns contain two distinct formats of signals. The project tagline contains semantic concepts, while variables like funding goal, category, target country, and campaign duration represent logistical launch constraints. If we only classified based on text, we would miss the financial feasibility; if we only classified on numbers, we would miss the concept. Stacking them together via feature fusion provides a comprehensive representation.

### Q: Why are the embeddings 384-dimensional?
* **Answer**: The embedding model `all-MiniLM-L6-v2` is configured to output 384-dimensional dense vectors. It represents the optimal trade-off between performance (retaining strong sentence semantic meaning) and efficiency (smaller disk footprint, lower RAM consumption, and faster cosine similarity computation compared to 768-dimensional models like BERT-base).

### Q: How are categorical variables handled?
* **Answer**: Categorical variables (`country`, `currency`, `category`, `main_category`) are encoded using a fitted scikit-learn `OneHotEncoder(handle_unknown="ignore")`. This converts each category value into separate binary sparse columns, yielding 225 columns. Using `handle_unknown="ignore"` ensures that if the system encounters an unseen category during inference, it ignores it instead of throwing a key error.

---

## 3. XGBoost Success Predictor

### Q: Why did you choose XGBoost over a Deep Neural Network?
* **Answer**: XGBoost is the standard for tabular classification tasks. Gradient-boosted decision trees naturally handle mixed sparse/dense variables, are robust to scaling variations, and handle collinear features better than neural networks. Additionally, XGBoost runs faster and requires far less hyperparameter tuning on CPUs.

### Q: What is the training and evaluation split?
* **Answer**: The dataset of 331,675 projects was split into a **80% training** and **20% testing** partition using stratified sampling. Stratification ensures that the ~35% base success rate is represented identically in both sets, preventing class distribution bias in our validation accuracy score of **93.56%**.

---

## 4. SHAP Explainability

### Q: What is SHAP?
* **Answer**: SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the outputs of machine learning models. It connects optimal credit allocation with local explanations by calculating the Shapley value of each feature, representing its contribution to the final prediction log-odds compared to the average model baseline.

### Q: What do positive and negative SHAP values mean?
* **Answer**: In our binary success classifier, the model predicts the probability of success. A positive SHAP value indicates that a feature *increases* the predicted probability of success (pushes it toward 1). A negative SHAP value indicates that a feature *decreases* the predicted probability of success (pushes it toward 0, favoring failure).

### Q: Why is SHAP not causal?
* **Answer**: SHAP measures feature importance based on correlation and feature dependencies in the trained model. If a feature has a high SHAP value (like `Number of Backers`), it indicates that the model heavily relies on that feature to classify success. It does not prove that changing that variable will physically cause the campaign to succeed (e.g. adding fake backers will not make a bad product succeed).

---

## 5. Local LLM & Grounding

### Q: Why did you choose local Ollama (Llama 3.2) over commercial APIs?
* **Answer**: Local inference via Ollama completely eliminates API key management, cost bottlenecks, and external network dependencies. It ensures that the system is 100% free and runnable offline, protecting data privacy and simplifying setup for recruiters or developers running the codebase locally.

### Q: How do you prevent LLM hallucinations and enforce grounding?
* **Answer**: 
  1. **Strict Decoupling**: Python handles all data calculations (risk, funding, SHAP, prediction). The LLM is never allowed to compute numbers.
  2. **JSON Facts Serialization**: Factual outputs are converted to a clean JSON string and passed to the prompt.
  3. **Prompts Boundaries**: The system prompt instructs the model to act as a qualitative strategist only, enforcing that all figures must match the JSON facts exactly.
  4. **Validation-Retry Control Loop**: A python class audits the LLM's sentence outputs using regular expressions. If it detects a contradiction, it resubmits the prompt once with specific correction guidelines, forcing Llama 3.2 to self-correct.

---

## 6. System & Software Engineering

### Q: How did you solve the macOS silent segmentation fault?
* **Answer**: On macOS, loading Sentence Transformers (which uses PyTorch) and then loading the pickled XGBoost classifier (via Joblib) caused an immediate crash. This is a known OpenMP conflict where both compiled C-libraries attempt to initialize their threading environments (`libomp` vs `libiomp5`). We solved this by importing `xgboost` at the absolute top of our entrypoint script, which allows the threading libraries to load compatibly.

### Q: How did you handle 500MB+ model/vector files in Git?
* **Answer**: Committing huge files violates GitHub's 100MB limits. We removed them from Git tracking using `git rm --cached` while keeping them locally intact for testing. We then configured `.gitignore` to protect the repo from tracking `data/embeddings/`, `artifacts/fusion/`, and `artifacts/chromadb/`, and documented the setup instructions in the README.

### Q: How is the repository made reproducible?
* **Answer**: 
  1. Frozen dependencies are documented in `requirements.txt` with pinned versions matching the local runtime.
  2. Persistent vector database caches are ignored but the script structures to generate/fetch them are provided.
  3. Local LLM operations run via standard Ollama model strings, removing API-key setup steps.
