import re
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model not downloaded during run
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

analyzer = SentimentIntensityAnalyzer()

PROMOTIONAL_PHRASES = [
    "must visit", "best ever", "perfect hotel", "everyone should", 
    "highly recommended", "number one", "unbeatable", "amazing experience",
    "will definitely return", "best in the world", "absolutely perfect", "100% perfect"
]

EVIDENCE_MARKERS = [
    "because", "since", "therefore", "resulted in", "we found", "i saw",
    "i noticed", "for example", "specifically", "the reason", "due to"
]

PERSONAL_EXPERIENCE = [
    "i stayed", "i visited", "i booked", "i used", "i noticed", "in my experience",
    "after two days", "during my stay", "we stayed", "my husband and i", "my family"
]

ABSOLUTE_WORDS = ["always", "never", "everyone", "nobody", "perfect", "terrible", "worst", "best"]

class FeatureExtractor:
    def __init__(self):
        pass
        
    def extract_features(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty_features()
            
        doc = nlp(text)
        
        # 1. Basic Text Features
        char_count = len(text)
        words = [token.text for token in doc if not token.is_punct]
        word_count = len(words)
        sentences = list(doc.sents)
        sentence_count = max(1, len(sentences))
        
        avg_sentence_length = word_count / sentence_count
        avg_word_length = sum(len(w) for w in words) / max(1, word_count)
        
        unique_words = set([w.lower() for w in words])
        lexical_diversity = len(unique_words) / max(1, word_count)
        
        # 2. Specificity (NER, Numbers)
        num_count = sum(1 for token in doc if token.like_num)
        entity_count = len(doc.ents)
        specificity_score = min(1.0, (num_count + entity_count) / max(1, sentence_count * 2))
        
        # 3. Personal Experience
        text_lower = text.lower()
        personal_count = sum(1 for phrase in PERSONAL_EXPERIENCE if phrase in text_lower)
        first_person = sum(1 for token in doc if token.lower_ in ["i", "me", "my", "we", "our", "us"])
        personal_experience_score = min(1.0, (personal_count * 2 + first_person) / max(1, word_count / 10))
        
        # 4. Evidence
        evidence_count = sum(1 for phrase in EVIDENCE_MARKERS if phrase in text_lower)
        evidence_score = min(1.0, evidence_count / max(1, sentence_count / 2))
        
        # 5. Sentiment
        vader_scores = analyzer.polarity_scores(text)
        polarity = vader_scores['compound']
        sentiment_label = "positive" if polarity >= 0.05 else ("negative" if polarity <= -0.05 else "neutral")
        if vader_scores['neu'] < 0.5 and (vader_scores['pos'] > 0.3 and vader_scores['neg'] > 0.3):
             sentiment_label = "mixed"
             
        # 6. Promotional Language
        promo_count = sum(1 for phrase in PROMOTIONAL_PHRASES if phrase in text_lower)
        promotional_language_score = min(1.0, promo_count / max(1, sentence_count / 3))
        
        # 7. Exaggeration
        superlative_count = sum(1 for token in doc if token.tag_ in ["JJS", "RBS"])
        absolute_count = sum(1 for w in words if w.lower() in ABSOLUTE_WORDS)
        exclamation_count = text.count('!')
        repeated_punct = len(re.findall(r'([!?.])\1+', text))
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        exaggeration_score = min(1.0, (superlative_count + absolute_count + repeated_punct + caps_words/2) / max(1, sentence_count))
        
        # 8. Linguistic Quality
        linguistic_quality = min(1.0, lexical_diversity + (1.0 if 10 < avg_sentence_length < 25 else 0.5)) / 2
        
        # 9. Style Features
        uppercase_ratio = caps_words / max(1, word_count)
        question_mark_count = text.count('?')
        repetition_ratio = 1.0 - lexical_diversity
        
        features = {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity,
            "repetition_ratio": repetition_ratio,
            "specificity_score": round(specificity_score, 4),
            "personal_experience_score": round(personal_experience_score, 4),
            "evidence_score": round(evidence_score, 4),
            "sentiment_polarity": round(polarity, 4),
            "sentiment_label": sentiment_label,
            "promotional_language_score": round(promotional_language_score, 4),
            "exaggeration_score": round(exaggeration_score, 4),
            "linguistic_quality": round(linguistic_quality, 4),
            "uppercase_ratio": round(uppercase_ratio, 4),
            "exclamation_count": exclamation_count,
            "question_mark_count": question_mark_count,
            "repeated_punct": repeated_punct
        }
        
        return features
        
    def _empty_features(self):
        return {
            "char_count": 0, "word_count": 0, "sentence_count": 0, "avg_sentence_length": 0.0,
            "avg_word_length": 0.0, "lexical_diversity": 0.0, "repetition_ratio": 0.0,
            "specificity_score": 0.0, "personal_experience_score": 0.0, "evidence_score": 0.0,
            "sentiment_polarity": 0.0, "sentiment_label": "neutral", "promotional_language_score": 0.0,
            "exaggeration_score": 0.0, "linguistic_quality": 0.0, "uppercase_ratio": 0.0,
            "exclamation_count": 0, "question_mark_count": 0, "repeated_punct": 0
        }
