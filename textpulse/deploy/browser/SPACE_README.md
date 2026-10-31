---
title: TextPulse
emoji: 📡
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: Real-time spam and sentiment on 1.6M real messages
---

# TextPulse

Real-time spam detection and sentiment analysis, trained on 1.64 million real messages, with a live dashboard.

- **Spam detector:** 97.95% accuracy on 7,133 unseen emails and SMS (Enron-Spam + SMS Spam Collection)
- **Sentiment analyser:** 82.30% accuracy on 320,000 unseen tweets (Sentiment140)

The stream replays held-out test messages the models never saw during training, classifies each one live, and shows the words that drove every prediction. You can also analyse your own text.

The two trained scikit-learn models run **directly in your browser**: they are exported to JSON (`scripts/export_for_browser.py`) and `engine.js` repeats the exact TF-IDF and logistic regression maths in JavaScript, agreeing with scikit-learn on 100% of the 6,000 test messages it was checked against. Each visitor gets their own stream, and nothing you type leaves your browser.

Source code, tests and a full code walkthrough: [github.com/sgoel2718/AI-ML-Projects](https://github.com/sgoel2718/AI-ML-Projects/tree/main/textpulse)

**Data:** Sentiment140 (Go, Bhayani & Huang, 2009) · SMS Spam Collection (Almeida & Hidalgo, UCI) · Enron-Spam (Metsis et al., 2006, preprocessed by M. Wiechmann).

Built by Shubham as part of the PG Certification in AI/ML at IIIT Hyderabad.
