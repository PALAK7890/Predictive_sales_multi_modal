"""
Build structured prompts for the LLM Campaign Consultant.
"""

import json


class PromptBuilder:
    """
    Constructs detailed, factual prompts for Ollama based on a structured
    facts dictionary from model predictions, explanations, and retrieval.
    """

    def build_prompt(self, facts: dict) -> str:
        # Convert the facts dictionary to a formatted JSON string for clear LLM parsing
        facts_json = json.dumps(facts, indent=2)

        prompt = f"""You are an AI campaign strategy consultant.

You are given VERIFIED STRUCTURED DATA produced by deterministic machine-learning and retrieval components:

```json
{facts_json}
```

Treat the supplied data as immutable facts. Your job is ONLY to explain and summarize these facts clearly, professionally, and qualitatively in the "AI CONSULTANT VERDICT" section.

STRICT GROUNDING RULES:
1. Never invent numerical values.
2. Never invent categories.
3. Never invent countries.
4. Never invent SHAP features.
5. Never invent campaign statistics.
6. Never invent similarity scores.
7. Never invent funding goals.
8. Never modify the supplied success probability.
9. Never modify the supplied risk score or risk level.
10. Never modify the supplied recommended funding goal.
11. Never describe a business recommendation as a SHAP feature.
12. Never create a SHAP impact that was not supplied.
13. Never introduce information that does not exist in the structured data.
14. If information is unavailable, explicitly say: "Not available from the analysis."
15. Do not infer missing numerical values.
16. Do not reconcile conflicting values by guessing.
17. Use the exact values supplied by the analysis.

SHAP RULE:
Only features/factors explicitly present inside the SHAP section of the structured JSON data may be described as SHAP factors or feature importance drivers.
Business recommendations, risk reasons, retrieval statistics, and general historical observations are NOT SHAP features.

NUMERICAL RULE:
All percentages, dollar amounts, scores, counts, and similarity values must come directly from the supplied structured data.
You may explain what these values mean qualitatively, but you must NOT change them.
For example, if the recommended funding goal is $29,711, you must not refer to it as $15,000 or any other number.

Your output is a strategic interpretation of the supplied analysis, not a new prediction.

Write your response with the following qualitative sections exactly:
- CAMPAIGN ASSESSMENT: Assess the campaign concept, category, and target country based on the data.
- SUCCESS PREDICTION INTERPRETATION: Explain qualitatively what the predicted outcome and probabilities mean.
- RISK INTERPRETATION: Discuss the risk level, risk score, and the reasons.
- FUNDING STRATEGY: Interpret the recommended goal and how it relates to the user's proposed goal.
- HISTORICAL CAMPAIGN INSIGHTS: Analyze the similarities and states of the retrieved campaigns.
- STRATEGIC RECOMMENDATIONS: Synthesize the recommendations given in the data.
- CONCRETE NEXT STEPS: Detail actionable next steps to minimize risk and improve success.

Do NOT include any markdown table or numerical overview list at the top that duplicates the raw data report. Focus purely on narrative interpretation.
"""
        return prompt
