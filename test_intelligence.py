from pprint import pprint

from src.explainability.explanation_engine import ExplanationEngine

engine = ExplanationEngine()

report = engine.explain_prediction(0)

pprint(report)