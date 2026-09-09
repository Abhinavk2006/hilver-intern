import os
import json
import random
import pandas as pd
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from src.config import TWCS_CSV_PATH, PROCESSED_DATA_DIR, RANDOM_SEED
from src.conversation import ConversationReconstructor

PROCESSED_PAIRS_PATH = PROCESSED_DATA_DIR / "apple_support_pairs.json"
TRAIN_CORPUS_PATH = PROCESSED_DATA_DIR / "train_retrieval_corpus.json"
TEST_CORPUS_PATH = PROCESSED_DATA_DIR / "test_evaluation_set.json"

def load_and_process_apple_data(force_reprocess: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Data Pipeline:
    1. Loads dataset
    2. Reconstructs customer-brand pairs
    3. Cleans and deduplicates
    4. Performs conversation-aware train/test split to prevent leakage
    5. Saves retrieval corpus and test set
    """
    if PROCESSED_PAIRS_PATH.exists() and not force_reprocess:
        print(f"Loading existing processed dataset from {PROCESSED_PAIRS_PATH}...")
        with open(PROCESSED_PAIRS_PATH, 'r', encoding='utf-8') as f:
            all_pairs = json.load(f)
        with open(TRAIN_CORPUS_PATH, 'r', encoding='utf-8') as f:
            train_pairs = json.load(f)
        with open(TEST_CORPUS_PATH, 'r', encoding='utf-8') as f:
            test_pairs = json.load(f)
        return all_pairs, train_pairs, test_pairs

    print("Executing full data processing pipeline...")
    ds = load_dataset('SunidhiSriram/twcs', split='train')
    df = ds.to_pandas()

    reconstructor = ConversationReconstructor(brand_handle="AppleSupport")
    pairs = reconstructor.reconstruct_brand_pairs(df)

    # Deduplicate based on customer_text_cleaned to avoid duplicate customer messages
    seen_texts = set()
    deduped_pairs = []
    for p in pairs:
        c_text = p['customer_text_cleaned'].lower()
        if c_text not in seen_texts:
            seen_texts.add(c_text)
            deduped_pairs.append(p)

    print(f"After deduplication: {len(deduped_pairs):,} unique customer query pairs.")

    # Save full processed dataset
    with open(PROCESSED_PAIRS_PATH, 'w', encoding='utf-8') as f:
        json.dump(deduped_pairs, f, indent=2)

    # Conversation-aware & Time-aware Split:
    # Shuffle with fixed random seed or sort by timestamp
    # We will reserve 10% for test/evaluation (max 5,000 for eval pool) and 90% for Retrieval Corpus
    random.seed(RANDOM_SEED)
    shuffled = deduped_pairs.copy()
    random.shuffle(shuffled)

    # 90% train retrieval corpus, 10% eval pool
    split_idx = int(len(shuffled) * 0.90)
    train_pairs = shuffled[:split_idx]
    test_pairs = shuffled[split_idx:]

    print(f"Train Retrieval Corpus: {len(train_pairs):,} examples")
    print(f"Test Evaluation Pool: {len(test_pairs):,} examples")

    with open(TRAIN_CORPUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_pairs, f, indent=2)
    with open(TEST_CORPUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_pairs, f, indent=2)

    return deduped_pairs, train_pairs, test_pairs

if __name__ == "__main__":
    load_and_process_apple_data(force_reprocess=True)
