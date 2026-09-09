import numpy as np
import pickle
from typing import Dict, Any, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.intents import INTENT_TAXONOMY, get_intent_list
from src.config import RESULTS_DIR

MODEL_SAVE_PATH = RESULTS_DIR / "tfidf_intent_model.pkl"

class MajorityIntentClassifier:
    """Baseline 1: Predicts majority class ('SOFTWARE_UPDATE_ISSUES') for all inputs."""
    def __init__(self, majority_class: str = "SOFTWARE_UPDATE_ISSUES"):
        self.majority_class = majority_class

    def predict(self, text: str) -> Tuple[str, float]:
        return self.majority_class, 1.0

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [(self.majority_class, 1.0) for _ in texts]


class TFIDFIntentClassifier:
    """Baseline 2 & Main Intent Classifier: TF-IDF + Logistic Regression."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42)
        self.is_trained = False

    def train(self, train_texts: List[str], train_labels: List[str]):
        print(f"Training TF-IDF + Logistic Regression Intent Classifier on {len(train_texts):,} examples...")
        X_tr = self.vectorizer.fit_transform(train_texts)
        self.classifier.fit(X_tr, train_labels)
        self.is_trained = True
        print("Classifier training complete.")

        # Save model
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump((self.vectorizer, self.classifier), f)

    def load_model(self) -> bool:
        if MODEL_SAVE_PATH.exists():
            with open(MODEL_SAVE_PATH, 'rb') as f:
                self.vectorizer, self.classifier = pickle.load(f)
            self.is_trained = True
            return True
        return False

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.is_trained:
            if not self.load_model():
                raise RuntimeError("Classifier is not trained!")
        
        vec = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(vec)[0]
        max_idx = int(np.argmax(probs))
        predicted_class = str(self.classifier.classes_[max_idx])
        confidence = float(probs[max_idx])
        return predicted_class, confidence

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        if not self.is_trained:
            if not self.load_model():
                raise RuntimeError("Classifier is not trained!")
        
        vecs = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(vecs)
        results = []
        classes = self.classifier.classes_
        for row in probs:
            max_idx = int(np.argmax(row))
            results.append((str(classes[max_idx]), float(row[max_idx])))
        return results
