import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.config import PROCESSED_DATA_DIR, GOLDEN_SET_DIR, RANDOM_SEED
from src.intents import INTENT_TAXONOMY, get_intent_list

GOLDEN_SET_FILE = GOLDEN_SET_DIR / "golden_set.json"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_evaluation_set.json"

def classify_heuristic(text: str) -> Tuple[str, str]:
    text_lower = text.lower()
    for intent_name, meta in INTENT_TAXONOMY.items():
        if intent_name == "OTHER_AMBIGUOUS_COMPLAINT":
            continue
        if any(re.search(pat, text_lower) for pat in meta.keywords):
            return intent_name, meta.default_handling
    return "OTHER_AMBIGUOUS_COMPLAINT", "ESCALATE"

def create_golden_set():
    print(f"Loading test evaluation pool from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_pool = json.load(f)

    random.seed(RANDOM_SEED)

    # Group test items by matched heuristic intent
    intent_buckets = {intent: [] for intent in get_intent_list()}
    
    for item in test_pool:
        intent, default_handling = classify_heuristic(item['customer_text_cleaned'])
        item_copy = dict(item)
        item_copy['candidate_intent'] = intent
        item_copy['candidate_handling'] = default_handling
        intent_buckets[intent].append(item_copy)

    print("\nTest pool bucket counts:")
    for intent, items in intent_buckets.items():
        print(f"  - {intent:32s}: {len(items):,}")

    # Stratified Sampling: Aim for 18-22 samples per intent category to total 200 examples
    golden_examples = []

    target_per_intent = 18
    for intent, items in intent_buckets.items():
        random.shuffle(items)
        selected = items[:target_per_intent]
        
        for idx, item in enumerate(selected):
            c_text = item['customer_text_cleaned']
            b_text = item['brand_text_cleaned']

            # Assign difficulty based on length & ambiguity
            if intent == "OTHER_AMBIGUOUS_COMPLAINT":
                difficulty = "hard" if len(c_text) > 80 else "medium"
            elif any(k in c_text.lower() for k in ["restart", "drain", "icloud", "battery"]):
                difficulty = "easy"
            else:
                difficulty = "medium"

            # Determine reference resolution key points
            reference_points = []
            if "http" in b_text:
                reference_points.append("Provide official Apple Support troubleshooting link")
            if "dm" in b_text.lower() or "direct message" in b_text.lower():
                reference_points.append("Escalate / Request Direct Message for account/device inspection")
            if not reference_points:
                reference_points.append("Provide standard troubleshooting steps")

            golden_entry = {
                "id": f"gold_{len(golden_examples) + 1:03d}",
                "conversation_id": item['conversation_id'],
                "customer_tweet_id": item['customer_tweet_id'],
                "customer_message": c_text,
                "gold_intent": intent,
                "gold_handling_decision": item['candidate_handling'],
                "reference_reply": b_text,
                "reference_resolution_points": reference_points,
                "difficulty": difficulty,
                "gold_escalation_reason": "Query requires account/billing inspection or hardware service" if item['candidate_handling'] == "ESCALATE" else "Standard technical query safely solvable via public KB guidance"
            }
            golden_examples.append(golden_entry)

    # Fill remaining to reach exactly 200 examples using diverse examples
    if len(golden_examples) < 200:
        remaining_needed = 200 - len(golden_examples)
        extra_pool = [i for i in test_pool if i['conversation_id'] not in {g['conversation_id'] for g in golden_examples}]
        random.shuffle(extra_pool)
        for item in extra_pool[:remaining_needed]:
            intent, default_handling = classify_heuristic(item['customer_text_cleaned'])
            golden_entry = {
                "id": f"gold_{len(golden_examples) + 1:03d}",
                "conversation_id": item['conversation_id'],
                "customer_tweet_id": item['customer_tweet_id'],
                "customer_message": item['customer_text_cleaned'],
                "gold_intent": intent,
                "gold_handling_decision": default_handling,
                "reference_reply": item['brand_text_cleaned'],
                "reference_resolution_points": ["Provide standard guidance"],
                "difficulty": "medium",
                "gold_escalation_reason": "Query requires private verification or service" if default_handling == "ESCALATE" else "Standard technical support guidance"
            }
            golden_examples.append(golden_entry)

    print(f"\nConstructed Golden Evaluation Set with {len(golden_examples)} entries.")
    
    with open(GOLDEN_SET_FILE, 'w', encoding='utf-8') as f:
        json.dump(golden_examples, f, indent=2)

    print(f"Saved Golden Set to {GOLDEN_SET_FILE}")

if __name__ == "__main__":
    create_golden_set()
