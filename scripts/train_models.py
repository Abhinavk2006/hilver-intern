import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.classifier import TFIDFIntentClassifier
from src.retriever import HistoricalRetriever
from scripts.create_golden_set import classify_heuristic

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_retrieval_corpus.json"

def main():
    print(f"Loading training retrieval corpus from {TRAIN_DATA_PATH}...")
    start_time = time.time()
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_corpus = json.load(f)
    print(f"Loaded {len(train_corpus):,} training examples in {time.time() - start_time:.2f}s.")

    # 1. Generate Silver Intent Labels for Classifier Training
    print("\nGenerating silver intent labels for training corpus using keyword heuristics...")
    train_texts = []
    train_labels = []
    
    for entry in train_corpus:
        text = entry['customer_text_cleaned']
        intent, _ = classify_heuristic(text)
        train_texts.append(text)
        train_labels.append(intent)

    # 2. Train TF-IDF + Logistic Regression Intent Classifier
    print("\n--- Training Intent Classifier ---")
    classifier = TFIDFIntentClassifier()
    classifier.train(train_texts, train_labels)

    # 3. Build Historical Retrieval Index
    print("\n--- Building Historical Retrieval Index ---")
    retriever = HistoricalRetriever(top_k=3)
    retriever.build_index(train_corpus)

    print("\n[SUCCESS] Model training and retrieval index construction complete!")

if __name__ == "__main__":
    main()
