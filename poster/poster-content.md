# TRUSTLENS

**An Explainable NLP Framework for Assessing the Trustworthiness of Online Reviews**

## Problem
Deceptive online reviews significantly impact consumer decisions. While black-box ML models can detect deception with high accuracy, they fail to explain their reasoning to end-users, reducing practical utility.

## Research Gap
Existing online review analysis approaches commonly focus on deception detection, sentiment analysis, or review helpfulness independently. There is comparatively less emphasis on combining multiple linguistic and review-quality signals into an explainable trustworthiness assessment.

## Objectives
- Extract transparent linguistic features from reviews.
- Train robust ML models on the Ott Corpus.
- Generate an explainable, multi-dimensional Trust Score.

## Dataset
**Ott Deceptive Opinion Spam Corpus**
- Total reviews: 1,596 (after duplicate removal)
- Truthful reviews: 796
- Deceptive reviews: 800

## Methodology & Architecture
Review Text → NLP Preprocessing (spaCy/NLTK) → Feature Extraction → TF-IDF Vectorization → Machine Learning Classification (Linear SVM) → Trust Score Engine → Explainability Dashboard

## Feature Engineering
Extracted numerical features include:
- Specificity (Named Entities, Numbers)
- Personal Experience (First-person pronouns)
- Evidence (Observation markers)
- Exaggeration (Superlatives, absolute words)
- Promotional Language
- Sentiment Polarity (VADER)

## Machine Learning Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 82.19% | 79.77% | 86.25% | 82.88% | 89.13% |
| **Linear SVM** | **86.56%** | **84.62%** | **89.38%** | **86.93%** | **93.30%** |
| Random Forest | 83.44% | 83.23% | 83.75% | 83.49% | 91.14% |

## Trust Score Framework
`Trust Score = Base + Positive Signals - Negative Signals - ML Signal`
The score dynamically bounds to 0-100 and outputs a Trust Level (Low/Moderate/High) along with positive/negative contributors.

## Conclusion
TrustLens successfully combines classical NLP techniques with supervised learning to deliver not just a prediction, but an interpretable evaluation of a review's textual trustworthiness.

## Technology Stack
- **Backend:** Python, FastAPI, Scikit-Learn, spaCy
- **Frontend:** React, Vite, Tailwind CSS, Recharts
- **Database:** SQLite

## References
- Ott, M., et al. (2011). Finding deceptive opinion spam by any stretch of the imagination. ACL 2011.
