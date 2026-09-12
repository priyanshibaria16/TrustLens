def calculate_trust_score(features: dict, ml_deception_prob: float) -> dict:
    """
    Calculates the Trust Score based on the transparent Trust Score Framework.
    
    Formula conceptually:
    Trustworthiness Score = Base + Positive Signals - Negative Signals - ML Signal
    
    The weights are framework-defined and are not claimed to represent
    universal probabilities of truth.
    """
    
    # Positive signals
    spec = features.get('specificity_score', 0)
    exp = features.get('personal_experience_score', 0)
    evid = features.get('evidence_score', 0)
    ling = features.get('linguistic_quality', 0)
    
    pos_score = (spec * 15) + (exp * 15) + (evid * 10) + (ling * 10)
    
    # Negative signals
    promo = features.get('promotional_language_score', 0)
    exag = features.get('exaggeration_score', 0)
    
    neg_score = (promo * 15) + (exag * 15) + (ml_deception_prob * 20)
    
    # Calculate raw score
    raw_score = 50 + pos_score - neg_score
    
    # Bound to 0-100
    trust_score = max(0, min(100, int(round(raw_score))))
    
    # Trust Level
    if trust_score <= 40:
        trust_level = "Low"
    elif trust_score <= 70:
        trust_level = "Moderate"
    else:
        trust_level = "High"
        
    # Generate Explanation
    pos_contributors = []
    if spec > 0.5: pos_contributors.append("High specificity and concrete details")
    if exp > 0.5: pos_contributors.append("Personal experience indicators detected")
    if evid > 0.5: pos_contributors.append("Supporting observations and evidence present")
    if ling > 0.7: pos_contributors.append("Strong linguistic consistency and quality")
    
    neg_contributors = []
    if promo > 0.3: neg_contributors.append("Excessive promotional language")
    if exag > 0.3: neg_contributors.append("Strong exaggeration and superlatives")
    if ml_deception_prob > 0.6: neg_contributors.append("Stylistic patterns consistent with deceptive reviews")
    
    # Base explanation text
    if trust_level == "High":
        exp_text = "This review received a high textual trustworthiness score because it contains concrete details, first-person experience, and supporting observations. It also contains relatively limited promotional language and exaggeration."
    elif trust_level == "Moderate":
        exp_text = "This review received a moderate textual trustworthiness score. While it contains some genuine review characteristics, it may lack sufficient concrete details or exhibits mild promotional or exaggerated language."
    else:
        exp_text = "This review received a lower textual trustworthiness score because it exhibits highly promotional language, exaggeration, or stylistic patterns often associated with deceptive reviews, while lacking concrete supporting details."

    return {
        "trust_score": trust_score,
        "trust_level": trust_level,
        "positive_contributors": pos_contributors,
        "negative_contributors": neg_contributors,
        "explanation": exp_text
    }
