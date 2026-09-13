import joblib
import os
import json
import numpy as np
import uuid
from scipy.sparse import hstack, csr_matrix

from ..nlp.feature_extraction import FeatureExtractor
from ..nlp.trust_score import calculate_trust_score

MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "models")
)


class PredictionService:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.model = None
        self.tfidf_word = None
        self.tfidf_char = None
        self.scaler = None
        self.use_char_tfidf = False
        self.numeric_features = [
            "char_count", "word_count", "sentence_count", "avg_sentence_length",
            "avg_word_length", "lexical_diversity", "repetition_ratio",
            "specificity_score", "personal_experience_score", "evidence_score",
            "sentiment_polarity", "promotional_language_score", "exaggeration_score",
            "linguistic_quality", "uppercase_ratio", "exclamation_count",
            "question_mark_count", "repeated_punct"
        ]

        try:
            self.model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
            self.tfidf_word = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
            self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

            # Load char TF-IDF if available
            char_path = os.path.join(MODELS_DIR, "tfidf_char_vectorizer.joblib")
            if os.path.exists(char_path):
                self.tfidf_char = joblib.load(char_path)
                self.use_char_tfidf = True

            # Load feature config if available
            config_path = os.path.join(MODELS_DIR, "feature_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                self.use_char_tfidf = cfg.get("use_char_tfidf", self.use_char_tfidf)
                self.numeric_features = cfg.get("numeric_features", self.numeric_features)

            print("✅ Models loaded successfully.")
        except Exception as e:
            print(f"⚠️  Could not load models. Please run ml/scripts/train.py first. Error: {e}")

    def analyze_review(self, text: str) -> dict:
        features = self.extractor.extract_features(text)

        ml_deception_prob = 0.5
        confidence = 0.5

        if self.model and self.tfidf_word and self.scaler:
            try:
                X_word = self.tfidf_word.transform([text])
                X_num = [[features[f] for f in self.numeric_features]]
                X_num_scaled = self.scaler.transform(X_num)

                if self.use_char_tfidf and self.tfidf_char is not None:
                    X_char = self.tfidf_char.transform([text])
                    X_combined = hstack([X_word, X_char, csr_matrix(X_num_scaled)])
                else:
                    X_combined = hstack([X_word, csr_matrix(X_num_scaled)])

                probs = self.model.predict_proba(X_combined)[0]
                ml_deception_prob = float(probs[1])
                confidence = float(max(probs))
            except Exception as e:
                print(f"Prediction error: {e}")

        trust_result = calculate_trust_score(features, ml_deception_prob)

        return {
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


prediction_service = PredictionService()
