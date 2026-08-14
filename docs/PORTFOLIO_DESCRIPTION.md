# Portfolio & LinkedIn Descriptions - Kickstarter Campaign Intelligence System

Use these pre-written descriptions for your LinkedIn project section, personal portfolio website, or to practice your 60-second interview elevator pitch.

---

## 1. Short Description (2–3 Sentences)
*Perfect for a quick project card on a personal website or a LinkedIn project listing.*

An end-to-end Explainable AI decision-support system that predicts Kickstarter campaign success and generates grounded pre-launch strategies. It combines semantic retrieval (ChromaDB + Sentence Transformers) over 331K+ historical projects, a 93.56% accurate XGBoost classifier, SHAP explainability, and a locally running Llama 3.2 consultant. The system features a custom sentence-based validation retry layer to programmatically prevent LLM hallucinations.

---

## 2. Medium Description (1 Paragraph)
*Suitable for a standard project block in a portfolio.*

The Kickstarter Campaign Intelligence System is a hybrid machine learning and retrieval-augmented generation (RAG) system designed to help crowdfunding creators optimize their pre-launch parameters. By fusing character-level and semantic-level text embeddings with structured project features (categories, goal, duration), the system achieves a **93.56% classification accuracy** using a tuned XGBoost classifier. To make the predictions transparent, the system calculates Shapley values via SHAP, mapping technical features to human-readable drivers. Rather than relying on a simple LLM wrapper, the pipeline compiles all computations into a structured JSON facts object and queries a local Llama 3.2 instance to draft a qualitative business strategy. To guarantee accuracy, a custom regex validation layer audits the LLM output, triggering automated correction retries on any numerical contradictions.

---

## 3. Technical Description (For Portfolio Details Section)
*Suitable for a dedicated detail page in your portfolio.*

This project showcases a production-level integration of tabular machine learning, explainable AI, vector databases, and local large language model orchestration. 

**Key Technical Implementations**:
* **Dense RAG Retrieval**: Encoded 331,675 crowdfunding campaigns into a persistent ChromaDB database using a 384-dimensional Sentence Transformer (`all-MiniLM-L6-v2`), permitting conceptual similarity searches via cosine distance.
* **Feature Fusion & Classification**: Engineered an preprocessing pipeline that scales, imputes, and one-hot encodes launch variables, stacking them alongside text embeddings to form a 609-feature matrix used to train a high-generalization XGBoost Classifier.
* **Explainability Interface**: Integrated SHAP TreeExplainer to resolve black-box ensemble decisions, writing a custom index-to-label mapper that separates factors into positive and negative strategic forces.
* **Grounding Validation & Retry Control Loop**: Programmatically decoupled narrative reasoning from calculations. The LLM acts solely as a qualitative strategist. A sentence-based regex parser validates model statements against factual variables (probability, goals, risk scores), executing an automated retry loop to enforce self-correction upon detecting hallucinations.

---

## 4. 60-Second Interview Explanation (Elevator Pitch)
*Practice saying this in response to: "Tell me about one of your projects."*

> *"I engineered the **Kickstarter Campaign Intelligence System**, which is an explainable AI decision-support system designed to optimize campaign pre-launch strategy using historical campaign patterns.
> 
> The project combines semantic RAG retrieval over 331,000 campaigns using ChromaDB and Sentence Transformers with a 93.56% accurate XGBoost success classifier. To solve the 'black-box' nature of tree ensembles, I integrated SHAP to isolate and map the positive and negative feature importances for a user's specific campaign idea.
> 
> The most interesting engineering challenge was integrating a local Llama 3.2 model to act as a qualitative consultant. To prevent LLM hallucinations, I built a structured JSON facts interface that makes Python the source of truth, and implemented a sentence-based regex validation layer in Python. If the LLM generates a number or goal that contradicts the machine learning calculations, the system automatically detects it, highlights the discrepancy, and retries Ollama once to force self-correction. 
> 
> This architecture ensures the output is both strategically deep and numerically grounded."*
