"""
Report Generator for Kickstarter Campaign Intelligence.
"""

import re
from src.retrieval.campaign_retriever import CampaignRetriever
from src.recommendation.recommendation_engine import RecommendationEngine
from src.recommendation.funding_optimizer import FundingGoalOptimizer
from src.recommendation.risk_analyzer import RiskAnalyzer
from src.models.predict_xgboost import CampaignPredictor
from src.explainability.explanation_engine import ExplanationEngine
from src.consultant.prompt_builder import PromptBuilder
from src.consultant.llm_consultant import CampaignConsultant


class ReportGenerator:
    """
    Orchestrates the entire campaign analysis pipeline and generates
    a cohesive Kickstarter Campaign Intelligence Report.
    """

    def __init__(self, ollama_model: str = "llama3.2"):
        self.retriever = CampaignRetriever()
        self.recommender = RecommendationEngine()
        self.optimizer = FundingGoalOptimizer()
        self.risk_analyzer = RiskAnalyzer()
        self.predictor = CampaignPredictor()
        self.explanation_engine = ExplanationEngine()
        self.prompt_builder = PromptBuilder()
        self.consultant = CampaignConsultant(model=ollama_model)

    def _to_python(self, val):
        """
        Recursively converts NumPy types (like float32, int64) inside dictionaries
        and lists into Python-native scalars to ensure JSON serializability.
        """
        if val is None:
            return None
        if isinstance(val, dict):
            return {k: self._to_python(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self._to_python(v) for v in val]
        if hasattr(val, "item"):  # Matches numpy scalars
            try:
                return val.item()
            except Exception:
                pass
        return val

    def _validate_verdict(self, verdict: str, facts: dict) -> list[str]:
        """
        Sentence-based validation to verify that the LLM output does not contradict
        the deterministic pipeline's statistics.
        """
        errors = []

        # Split verdict into sentences
        sentences = re.split(r'[.!?\n]', verdict)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 1. Validate success probability
            is_success_prob = any(
                term in sentence.lower()
                for term in [
                    "success probability",
                    "probability of success",
                    "success rate",
                    "chance of success",
                ]
            )
            if is_success_prob:
                prob_matches = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*%', sentence)
                expected_success = facts["prediction"]["success_probability"]
                for match in prob_matches:
                    val = float(match)
                    if abs(val - expected_success) > 1.0:  # 1.0% tolerance
                        errors.append(
                            f"Sentence '{sentence}' mentions success probability of {val}%, which conflicts with actual success probability of {expected_success}%"
                        )

            # 2. Validate failure probability
            is_failure_prob = any(
                term in sentence.lower()
                for term in [
                    "failure probability",
                    "probability of failure",
                    "failure rate",
                    "chance of failure",
                ]
            )
            if is_failure_prob:
                prob_matches = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*%', sentence)
                expected_failure = facts["prediction"]["failure_probability"]
                for match in prob_matches:
                    val = float(match)
                    if abs(val - expected_failure) > 1.0:
                        errors.append(
                            f"Sentence '{sentence}' mentions failure probability of {val}%, which conflicts with actual failure probability of {expected_failure}%"
                        )

            # 3. Validate risk score
            is_risk_score = "risk score" in sentence.lower()
            if is_risk_score:
                numbers = re.findall(r'\b([0-9]{1,3})\b', sentence)
                expected_score = facts["risk"]["risk_score"]
                for num_str in numbers:
                    val = int(num_str)
                    if val != expected_score and val != 100:  # Allow '100' for fractions like 60/100
                        errors.append(
                            f"Sentence '{sentence}' mentions risk score of {val}, which conflicts with actual risk score of {expected_score}"
                        )

            # 4. Validate risk level
            is_risk_level = any(
                term in sentence.lower() for term in ["risk level", "risk is"]
            )
            if is_risk_level:
                expected_level = facts["risk"]["risk_level"]
                levels_found = re.findall(
                    r'\b(LOW|MEDIUM|HIGH)\b', sentence, re.IGNORECASE
                )
                for level in levels_found:
                    if level.upper() != expected_level.upper():
                        errors.append(
                            f"Sentence '{sentence}' mentions risk level '{level}', which conflicts with actual risk level '{expected_level}'"
                        )

            # 5. Validate recommended goal
            is_rec_goal = any(
                term in sentence.lower()
                for term in [
                    "recommended goal",
                    "recommended funding goal",
                    "recommend a goal",
                    "recommend funding goal",
                ]
            )
            if is_rec_goal:
                numbers = re.findall(r'\$?([0-9,]+(?:\.[0-9]+)?)', sentence)
                expected_rec = facts["funding"]["recommended_goal"]
                user_goal = facts["campaign"]["funding_goal"]
                for num_str in numbers:
                    try:
                        clean_val = float(num_str.replace(",", ""))
                        if clean_val < 100:  # Skip small numbers (like counters or durations)
                            continue
                        if expected_rec is not None and abs(clean_val - expected_rec) > 10.0:
                            if abs(clean_val - user_goal) < 10.0:
                                errors.append(
                                    f"Sentence '{sentence}' mistakenly recommends your proposed goal of ${clean_val:,.0f} instead of the ML recommended goal of ${expected_rec:,.0f}"
                                )
                            else:
                                errors.append(
                                    f"Sentence '{sentence}' mentions recommended goal of ${clean_val:,.0f}, which conflicts with actual recommended goal of ${expected_rec:,.0f}"
                                )
                    except ValueError:
                        pass

            # 6. Validate average goal of successes
            is_avg_goal = any(
                term in sentence.lower()
                for term in [
                    "average goal",
                    "average successful goal",
                    "average goal of successful campaigns",
                ]
            )
            if is_avg_goal:
                numbers = re.findall(r'\$?([0-9,]+(?:\.[0-9]+)?)', sentence)
                expected_avg = facts["funding"]["average_goal"]
                for num_str in numbers:
                    try:
                        clean_val = float(num_str.replace(",", ""))
                        if clean_val < 100:
                            continue
                        if expected_avg is not None and abs(clean_val - expected_avg) > 10.0:
                            errors.append(
                                f"Sentence '{sentence}' mentions average goal of ${clean_val:,.0f}, which conflicts with actual average goal of ${expected_avg:,.0f}"
                            )
                    except ValueError:
                        pass

        return errors

    def generate_report(
        self,
        title: str,
        category: str,
        main_category: str,
        country: str,
        currency: str,
        goal: float,
        duration: int = 30,
    ) -> str:
        """
        Runs the end-to-end analytics and ML tools and queries local Ollama
        to assemble the final comprehensive intelligence report.
        """
        print("\n[1/6] Retrieving similar campaigns...")
        retrieval_results = self.retriever.search(title, top_k=50)

        print("[2/6] Running recommendation engine & risk analysis...")
        recommendation_results = self.recommender.recommend(title, top_k=50)
        funding_results = self.optimizer.optimize(title, top_k=50)
        risk_results = self.risk_analyzer.analyze(title)

        print("[3/6] Running XGBoost success prediction model...")
        prediction_results = self.predictor.predict(
            title=title,
            category=category,
            main_category=main_category,
            country=country,
            currency=currency,
            goal=goal,
            duration=duration,
        )

        print("[4/6] Generating SHAP explanations...")
        X_sample = self.predictor.get_features(
            title=title,
            category=category,
            main_category=main_category,
            country=country,
            currency=currency,
            goal=goal,
            duration=duration,
        )
        explanation_results = self.explanation_engine.explain_instance(X_sample)

        # Assemble the facts dictionary
        raw_facts = {
            "campaign": {
                "title": title,
                "category": category,
                "main_category": main_category,
                "country": country,
                "currency": currency,
                "funding_goal": goal,
                "duration": duration
            },
            "prediction": {
                "predicted_outcome": prediction_results["prediction"],
                "success_probability": prediction_results["success_probability"],
                "failure_probability": prediction_results["failure_probability"]
            },
            "risk": {
                "risk_level": risk_results["risk_level"],
                "risk_score": risk_results["risk_score"],
                "success_rate": risk_results["success_rate"],
                "reasons": risk_results.get("reasons", []),
                "suggestions": risk_results.get("suggestions", [])
            },
            "funding": {
                "recommended_goal": funding_results.get("recommended_goal", 0),
                "average_goal": funding_results.get("average_goal", 0),
                "minimum_goal": funding_results.get("minimum_goal", 0),
                "maximum_goal": funding_results.get("maximum_goal", 0),
                "successful_campaigns": funding_results.get("successful_campaigns", 0),
                "total_compared": funding_results.get("total_compared", 0)
            },
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
                "positive_factors": [
                    {
                        "feature": f["feature"],
                        "impact": f["shap_value"]
                    }
                    for f in explanation_results.get("positive_factors", [])[:5]
                ],
                "negative_factors": [
                    {
                        "feature": f["feature"],
                        "impact": f["shap_value"]
                    }
                    for f in explanation_results.get("negative_factors", [])[:5]
                ]
            },
            "business_recommendations": recommendation_results.get("recommendations", [])
        }

        # Convert all NumPy types to Python scalars
        facts = self._to_python(raw_facts)

        print("[5/6] Building prompt and querying AI Campaign Consultant (Ollama)...")
        prompt = self.prompt_builder.build_prompt(facts)
        verdict = self.consultant.ask(prompt)

        # Validate verdict output
        validation_errors = self._validate_verdict(verdict, facts)
        if validation_errors:
            print(f"\n[Validation Warning] LLM output failed validation with issues: {validation_errors}")
            print("Retrying Ollama once with grounding corrections...")
            retry_prompt = prompt + (
                f"\n\n[SYSTEM NOTE: Your previous response contained the following factual contradictions with the data:\n"
                + "\n".join([f"- {err}" for err in validation_errors])
                + "\n\nPlease rewrite the response to correct these contradictions. Make sure every success percentage, risk score/level, and recommended funding goal matches the JSON data exactly.]"
            )
            verdict = self.consultant.ask(retry_prompt)
            validation_errors = self._validate_verdict(verdict, facts)

        # If it still fails validation, we inject a visual warning block in the output verdict
        if validation_errors:
            disclaimer = (
                f"> [!WARNING]\n"
                f"> **Grounding Validation Alert**: The AI-generated verdict below contained statements that contradicted the deterministic ML pipeline.\n"
                f"> The primary numerical pipeline above remains the authoritative source of truth.\n"
                f"> Discrepancies detected:\n"
            )
            for err in validation_errors:
                disclaimer += f"> - {err}\n"
            disclaimer += "\n"
            verdict = disclaimer + verdict

        print("[6/6] Formatting final intelligence report...")

        # Format details for deterministic sections
        risk_reasons = "\n".join([f"  - {r}" for r in risk_results.get("reasons", [])]) if risk_results.get("reasons") else "  - None flagged"
        risk_suggestions = "\n".join([f"  - {s}" for s in risk_results.get("suggestions", [])]) if risk_results.get("suggestions") else "  - None flagged"

        similar_campaigns_lines = []
        for c in retrieval_results[:5]:
            similar_campaigns_lines.append(
                f"- \"{c['title']}\" (Similarity: {c['similarity'] * 100:.1f}%)\n"
                f"  Category: {c['category']} | Main: {c['main_category']}\n"
                f"  Goal: {c['country']} {c['goal']:,.0f} | Pledged: {c['country']} {c['pledged']:,.0f} | State: {c['state']}"
            )
        similar_campaigns_str = "\n".join(similar_campaigns_lines)

        pos_factors_lines = []
        for f in explanation_results.get("positive_factors", [])[:5]:
            pos_factors_lines.append(f"  - {f['feature']} (SHAP Impact: +{f['shap_value']:.4f})")
        pos_factors_str = "\n".join(pos_factors_lines) if pos_factors_lines else "  - None identified"

        neg_factors_lines = []
        for f in explanation_results.get("negative_factors", [])[:5]:
            neg_factors_lines.append(f"  - {f['feature']} (SHAP Impact: {f['shap_value']:.4f})")
        neg_factors_str = "\n".join(neg_factors_lines) if neg_factors_lines else "  - None identified"

        business_recs = "\n".join([f"- {r}" for r in recommendation_results.get("recommendations", [])])

        # Construct final printed layout ensuring clear separation
        report = f"""==================================================
KICKSTARTER CAMPAIGN INTELLIGENCE REPORT
==================================================

DETERMINISTIC ANALYSIS
----------------------
Generated directly from the retrieval and ML pipeline.

Campaign Idea
Title/Description: {title}
Category: {category} (Main Category: {main_category})
Funding Goal: {currency} {goal:,.2f}
Duration: {duration} days
Country: {country}

--------------------------------------------------

SIMILAR CAMPAIGNS
{similar_campaigns_str}

--------------------------------------------------

SUCCESS PREDICTION
Predicted Outcome: {prediction_results['prediction']}
Success Probability: {prediction_results['success_probability']}%
Failure Probability: {prediction_results['failure_probability']}%

--------------------------------------------------

RISK ANALYSIS
Risk Level: {risk_results['risk_level']}
Risk Score: {risk_results['risk_score']}/100
Reasons:
{risk_reasons}
Suggestions:
{risk_suggestions}

--------------------------------------------------

FUNDING RECOMMENDATION
Recommended Goal: USD {funding_results.get('recommended_goal', 0):,.2f} (Median of successful similar campaigns)
Average Goal of Successes: USD {funding_results.get('average_goal', 0):,.2f}
Goal Range (Successful References): USD {funding_results.get('minimum_goal', 0):,.2f} to USD {funding_results.get('maximum_goal', 0):,.2f}
Successful Campaign References: {funding_results.get('successful_campaigns', 0)} out of {funding_results.get('total_compared', 0)} compared

--------------------------------------------------

MODEL EXPLANATION
Top Positive Factors (Driving Success):
{pos_factors_str}

Top Negative Factors (Driving Failure):
{neg_factors_str}

--------------------------------------------------

BUSINESS RECOMMENDATIONS
{business_recs}

==================================================
AI CONSULTANT VERDICT
==================================================
Natural-language interpretation of the deterministic analysis.

{verdict}
"""
        return report
