import json
import pickle
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR

INDEX_SAVE_PATH = RESULTS_DIR / "retrieval_index.pkl"

class HistoricalRetriever:
    """
    Retrieval engine indexing 94,098 historical AppleSupport conversations.
    Uses TF-IDF + Cosine Similarity for fast, deterministic, leakage-free retrieval.
    """
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        self.corpus_pairs: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self.is_indexed = False

    def build_index(self, corpus_pairs: List[Dict[str, Any]]):
        print(f"Building retrieval index over {len(corpus_pairs):,} historical AppleSupport query-response pairs...")
        self.corpus_pairs = corpus_pairs
        texts = [p['customer_text_cleaned'] for p in corpus_pairs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_indexed = True

        # Save index
        with open(INDEX_SAVE_PATH, 'wb') as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix, self.corpus_pairs), f)
        print("Retrieval index constructed and saved successfully.")

    def load_index(self) -> bool:
        if INDEX_SAVE_PATH.exists():
            with open(INDEX_SAVE_PATH, 'rb') as f:
                self.vectorizer, self.tfidf_matrix, self.corpus_pairs = pickle.load(f)
            self.is_indexed = True
            return True
        return False

    def retrieve(self, query: str, top_k: int = None, exclude_tweet_id: int = None) -> List[Dict[str, Any]]:
        if not self.is_indexed:
            if not self.load_index():
                raise RuntimeError("Retrieval index is not built!")

        k = top_k if top_k is not None else self.top_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get top indices (fetching k+5 to allow filtering out query itself if leakage occurs)
        top_indices = np.argsort(scores)[::-1][:k + 5]

        results = []
        for idx in top_indices:
            candidate = self.corpus_pairs[idx]
            
            # Anti-leakage guard: Never allow an evaluation example to retrieve itself
            if exclude_tweet_id is not None and candidate.get('customer_tweet_id') == exclude_tweet_id:
                continue

            results.append({
                "conversation_id": candidate['conversation_id'],
                "customer_tweet_id": candidate['customer_tweet_id'],
                "customer_message": candidate['customer_text_cleaned'],
                "brand_response": candidate['brand_text_cleaned'],
                "similarity_score": round(float(scores[idx]), 4)
            })

            if len(results) == k:
                break

        return results
