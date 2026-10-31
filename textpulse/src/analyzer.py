"""
Use a trained model: predict a label and explain which words caused it.

    analyzer = TextAnalyzer(SPAM_MODEL_FILE)
    analyzer.analyze("Congratulations! You won a free prize, click here")
    -> {"label": "spam", "confidence": 0.99,
        "top_words": [{"word": "free", "weight": 1.21}, ...]}
"""

import joblib
import numpy as np


def word_weights(model):
    """
    One number per word: how strongly that word pushes toward the second
    class (spam / positive). Positive pushes toward it, negative pushes away.

    Both algorithms we train are "linear": the final score is just
    (sum of word value x word weight), so these weights explain the decision
    exactly. Day 3 called the Naive Bayes version the log-likelihood ratio.
    """
    if hasattr(model, "feature_log_prob_"):   # Naive Bayes
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]
    return model.coef_[0]                      # Logistic Regression


class TextAnalyzer:
    """Loads one saved model and analyses text with it."""

    def __init__(self, model_file):
        self.pipeline = joblib.load(model_file)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.model = self.pipeline.named_steps["model"]
        self.labels = list(self.model.classes_)            # e.g. ["ham", "spam"]
        self.words = self.vectorizer.get_feature_names_out()
        self.weights = word_weights(self.model)

    def analyze(self, text, top=4):
        """Predict one message and list the words that drove the decision."""
        numbers = self.vectorizer.transform([text])       # text -> TF-IDF numbers
        probabilities = self.model.predict_proba(numbers)[0]
        winner = int(probabilities.argmax())              # 0 or 1

        # A message only contains a few of the 200,000 words. The sparse
        # matrix stores just those: .indices = which words, .data = their values.
        word_ids = numbers.indices
        contributions = numbers.data * self.weights[word_ids]

        # For class 0 (ham / negative), words with NEGATIVE weights helped, so
        # flip the sign: "helped the winner" is always a positive number.
        direction = 1 if winner == 1 else -1
        helped = contributions * direction
        order = np.argsort(-helped)[:top]                # biggest helpers first

        top_words = [
            {"word": str(self.words[word_ids[i]]), "weight": round(float(helped[i]), 3)}
            for i in order
            if helped[i] > 0
        ]
        return {
            "label": self.labels[winner],
            "confidence": round(float(probabilities[winner]), 4),
            "top_words": top_words,
        }

    def predict_many(self, texts):
        """Fast path for scoring many messages at once (no explanations)."""
        return self.pipeline.predict(texts)
