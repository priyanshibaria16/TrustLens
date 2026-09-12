import joblib
import os
import numpy as np
import uuid
import json
from ..nlp.feature_extraction import FeatureExtractor
from ..nlp.trust_score import calculate_trust_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "models")

class PredictionService:
    def __init__(self):
        self.extractor = FeatureExtractor()
        try:
            self.model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
            self.tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
            self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        except Exception as e:
            print(f"Warning: Could not load models. Please train first. Error: {e}")
            self.model = None
            self.tfidf = None
            self.scaler = None
            
        self.numeric_features = [
            "char_count", "word_count", "sentence_count", "avg_sentence_length", 
            "avg_word_length", "lexical_diversity", "repetition_ratio",
            "specificity_score", "personal_experience_score", "evidence_score",
            "sentiment_polarity", "promotional_language_score", "exaggeration_score",
            "linguistic_quality", "uppercase_ratio", "exclamation_count",
            "question_mark_count", "repeated_punct"
        ]

    def analyze_review(self, text: str) -> dict:
        features = self.extractor.extract_features(text)
        
        ml_deception_prob = 0.5
        confidence = 0.5
        
        if self.model and self.tfidf and self.scaler:
            X_text_tfidf = self.tfidf.transform([text])
            X_num = [[features[f] for f in self.numeric_features]]
            X_num_scaled = self.scaler.transform(X_num)
            
            X_combined = np.hstack((X_text_tfidf.toarray(), X_num_scaled))
            
            # Predict probabilities
            probs = self.model.predict_proba(X_combined)[0]
            ml_deception_prob = float(probs[1]) # Probability of being deceptive (label 1)
            # Confidence is the max probability
            confidence = float(max(probs))
            
        trust_result = calculate_trust_score(features, ml_deception_prob)
        
        response = {
            "success": True,
            "review_id": str(uuid.uuid4()),
            "trust_score": trust_result["trust_score"],
            "trust_level": trust_result["trust_level"],
            "confidence": round(confidence, 2),
            "sentiment": {
                "label": features["sentiment_label"],
                "polarity": features["sentiment_polarity"]
            },
            "features": {
                "specificity": features["specificity_score"],
                "personal_experience": features["personal_experience_score"],
                "evidence": features["evidence_score"],
                "linguistic_quality": features["linguistic_quality"],
                "promotion": features["promotional_language_score"],
                "exaggeration": features["exaggeration_score"]
            },
            "positive_contributors": trust_result["positive_contributors"],
            "negative_contributors": trust_result["negative_contributors"],
            "explanation": trust_result["explanation"],
            "warnings": [
                "Textual trustworthiness does not establish factual truth."
            ]
        }
        return response

prediction_service = PredictionService()
