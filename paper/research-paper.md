# TrustLens: An Explainable NLP Framework for Assessing the Trustworthiness of Online Reviews

## 1. Abstract
The proliferation of deceptive online reviews has undermined consumer trust in e-commerce and hospitality platforms. We introduce TrustLens, an explainable Natural Language Processing (NLP) framework designed to assess the textual trustworthiness of online reviews. Using the Ott Deceptive Opinion Spam Corpus as the primary dataset, the system extracts linguistic features, semantic indicators, and deceptive stylistic patterns. An explainable machine learning approach based on TF-IDF and engineered numerical features achieves an F1-score of 86.93% using a Linear SVM classifier. The framework outputs a bounding 0-100 Trust Score accompanied by natural language explanations highlighting positive and negative contributors, ensuring transparent assessment.

## 2. Keywords
Deceptive Opinion Spam, Explainable AI, Natural Language Processing, Feature Engineering, Trustworthiness Assessment

## 3. Introduction
Online reviews significantly influence purchasing decisions. However, the ease of posting fraudulent feedback has led to "opinion spam." TrustLens aims to bridge the gap between black-box deception detection models and user-centric trust assessment by combining supervised machine learning with a transparent linguistic scoring framework.

## 4. Background & 5. Literature Review
Research in deceptive opinion spam prominently features the work by Ott et al. (2011), who compiled the first large-scale gold-standard dataset using crowdsourcing. Subsequent approaches have utilized n-gram models, psycholinguistic features (LIWC), and deep learning architectures. However, these systems primarily focus on binary classification rather than providing a holistic, explainable assessment of the review's textual signals.

## 6. Research Gap
Existing online review analysis approaches commonly focus on deception detection, sentiment analysis, or review helpfulness independently. There is comparatively less emphasis on combining multiple linguistic and review-quality signals into an explainable trustworthiness assessment that communicates the reasons behind the prediction to end users.

## 7. Objectives
- To develop a comprehensive feature extraction pipeline capturing specificity, personal experience, and stylistic exaggeration.
- To train and compare classical machine learning models on the Ott Corpus.
- To synthesize model predictions and explicit linguistic signals into an explainable Trust Score.

## 8. Proposed Methodology & 9. System Architecture
**System Architecture:** Review Text → NLP Preprocessing (spaCy/NLTK) → Feature Extraction (Sentiment, Specificity, Exaggeration) → TF-IDF & Scaled Numeric Vectorization → Machine Learning Classification (SVM/RF/LR) → Trust Score Framework → Frontend Dashboard.

## 10. Dataset
We utilized exclusively the Ott Deceptive Opinion Spam Corpus, consisting of approximately 1,600 hotel reviews. After duplicate removal, 1,596 reviews (796 truthful, 800 deceptive) remained, providing a balanced binary classification dataset. The labeled deception categories in the Ott corpus are used as supervised learning signals for modeling textual indicators associated with deceptive review writing.

## 11. Data Preprocessing & 12. Feature Engineering
Text preprocessing involved tokenization and POS tagging. Feature engineering extracted:
- **Specificity**: Named entities, numerical values.
- **Personal Experience**: First-person pronouns and specific experiential phrases.
- **Evidence**: Observation markers.
- **Stylistic Anomalies**: Promotional phrases, excessive capitalization, and repeated punctuation.

## 13. Machine Learning Models & 14. Trust Score Framework
We benchmarked Logistic Regression, Linear SVM, and Random Forest. 
The Trust Score Framework operates conceptually as:
`Trustworthiness Score = Base Score + Positive Linguistic Signals - Negative Stylistic Signals - ML Deception Signal`
This ensures the score is not merely the inverse of the deception probability, but a composite of observable qualities.

## 15. Explainability
TrustLens provides explicit positive and negative contributors based on the magnitude of the extracted feature scores, coupled with a dynamic natural language explanation (e.g., "This review received a high textual trustworthiness score because it contains concrete details and first-person experience").

## 16. Experimental Setup & 17. Results
Models were trained on an 80/20 stratified split. TF-IDF was fitted solely on the training data.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 82.19% | 79.77% | 86.25% | 82.88% | 89.13% |
| **Linear SVM** | **86.56%** | **84.62%** | **89.38%** | **86.93%** | **93.30%** |
| Random Forest | 83.44% | 83.23% | 83.75% | 83.49% | 91.14% |

The Linear SVM achieved the best overall performance, demonstrating that a combination of TF-IDF vectors (max 2000 features, unigrams+bigrams) and engineered linguistic features effectively separates the truthful and deceptive classes.

## 18. Discussion
The incorporation of specific linguistic features (exaggeration, promotional language) alongside TF-IDF representations allows the system to identify the nuanced differences between truthful and crowdsourced deceptive writing. 

## 19. Limitations
1. Ott is a relatively small dataset focusing exclusively on hotel reviews.
2. The system estimates textual trustworthiness, not factual truth.
3. Trust score thresholds are prototype heuristics and not universal probabilities.

## 20. Future Scope
Future work involves evaluating the framework across diverse domains (e.g., product or restaurant reviews), integrating large language models (LLMs) for deeper semantic extraction, and executing large-scale human evaluation on the effectiveness of the Trust Score explanations.

## 21. Conclusion
TrustLens successfully demonstrates an end-to-end explainable NLP framework. By leveraging the Ott Corpus, the system moves beyond binary classification to provide a transparent, multi-dimensional assessment of review trustworthiness, making AI-driven decisions interpretable for end-users.

## 22. References
- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.
