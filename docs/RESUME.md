# Professional Resume Bullets - Kickstarter Campaign Intelligence System

Below are three versions of pre-written resume bullet points summarizing the engineering depth of the project. Choose the version that best fits your resume layout.

---

### VERSION A — One-Line Format
*Suitable for a compact project listing section.*

* **Kickstarter Campaign Intelligence System**: Engineered an explainable AI decision-support pipeline combining semantic RAG retrieval (ChromaDB + Sentence Transformers) over 331K+ campaigns with a 93.56% accurate XGBoost success predictor, SHAP explainability, and a locally hosted LLM consultant (Ollama/Llama 3.2) featuring automated self-correction validation retries.

---

### VERSION B — Two-Bullet Format
*Suitable for a standard ML/Software Engineering resume.*

* **Engineered an End-to-End Decision Support System**: Fused 384-dimensional dense semantic embeddings (Sentence Transformers) with 225 preprocessed tabular features over a dataset of 331,675 projects; built an XGBoost classifier achieving **93.56% validation accuracy** to predict success and failure probabilities.
* **Integrated Explainability & Grounded Local Inference**: Extracted and mapped feature importances using SHAP TreeExplainer to clean human-readable categories; built an LLM consultant pipeline (Ollama + Llama 3.2) featuring a lightweight sentence-based validation retry layer to programmatically detect and self-correct LLM numerical contradictions.

---

### VERSION C — Three-Bullet Format
*Suitable for an AI/ML-heavy resume.*

* **Developed a Hybrid Retrieval & Fusion Pipeline**: Indexed 331,675 Kickstarter projects in a persistent ChromaDB vector store; optimized campaign discovery utilizing dense semantic embeddings (`all-MiniLM-L6-v2`) and cosine similarity, outperforming traditional keyword-matching search.
* **Built a High-Accuracy Classifier & XAI Explainer**: Stacked tabular inputs and embeddings into a 609-feature matrix to train an XGBoost classifier (**93.56% accuracy**); integrated SHAP TreeExplainer to calculate Shapley values, programmatically mapping technical indices to positive/negative success drivers.
* **Designed a Self-Correcting Local LLM Orchestrator**: Decoupled narrative reasoning from calculations by passing structured JSON facts to local Llama 3.2; engineered a custom validation layer in Python using sentence-based matching to scan for contradictions, triggering an automated correction-prompt retry loop on discrepancies.
