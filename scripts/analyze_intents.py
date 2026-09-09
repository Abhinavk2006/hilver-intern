import json
import re
from collections import Counter
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR

def analyze_intent_distribution():
    train_path = PROCESSED_DATA_DIR / "train_retrieval_corpus.json"
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    print(f"Analyzing {len(train_data):,} training query pairs...")

    # Define rule-based intent matching to discover empirical frequencies on AppleSupport data
    intent_keywords = {
        "SOFTWARE_UPDATE_ISSUES": [r'update', r'ios \d', r'ios1', r'upgrade', r'downgrade', r'beta'],
        "BATTERY_POWER_ISSUES": [r'battery', r'drain', r'charge', r'charging', r'overheat', r'die', r'dying'],
        "DEVICE_FREEZE_REBOOT": [r'restart', r'freeze', r'frozen', r'stuck', r'loop', r'black screen', r'boot', r'crash'],
        "ACCOUNT_APPLE_ID_SECURITY": [r'apple id', r'password', r'lock', r'locked', r'sign in', r'login', r'passcode', r'disabled', r'two factor', r'2fa'],
        "ICLOUD_STORAGE_SYNC": [r'icloud', r'storage', r'backup', r'sync', r'photos', r'cloud', r'full'],
        "BILLING_APP_STORE_SUBSCRIPTION": [r'charge', r'refund', r'billing', r'invoice', r'app store', r'subscription', r'apple pay', r'purchase', r'receipt'],
        "CONNECTIVITY_NETWORK_BLUETOOTH": [r'wifi', r'wi-fi', r'bluetooth', r'cellular', r'signal', r'service', r'connection', r'pair', r'airdrop'],
        "HARDWARE_DISPLAY_AUDIO": [r'screen', r'touch', r'display', r'crack', r'speaker', r'microphone', r'audio', r'sound', r'camera', r'headphone', r'jack'],
        "HARDWARE_REPAIR_SERVICE_INQUIRY": [r'store', r'appointment', r'genius', r'repair', r'warranty', r'replacement', r'cost', r'trade in'],
        "GENERAL_HOW_TO_INFO": [r'how to', r'how do i', r'feature', r'setting', r'settings', r'where is']
    }

    intent_counts = Counter()
    unclassified_samples = []

    for item in train_data:
        text = item['customer_text_cleaned'].lower()
        matched = False
        for intent, patterns in intent_keywords.items():
            if any(re.search(pat, text) for pat in patterns):
                intent_counts[intent] += 1
                matched = True
                break
        if not matched:
            intent_counts["OTHER_UNCLASSIFIED"] += 1
            if len(unclassified_samples) < 20:
                unclassified_samples.append(text)

    print("\n--- EMPIRICAL INTENT DISTRIBUTION ---")
    total = len(train_data)
    for intent, count in intent_counts.most_common():
        pct = (count / total) * 100
        print(f"  - {intent:32s}: {count:6,} ({pct:5.2f}%)")

    out_file = RESULTS_DIR / "intent_analysis.json"
    with open(out_file, "w") as f:
        json.dump({
            "total_analyzed": total,
            "distribution": dict(intent_counts),
            "unclassified_samples": unclassified_samples
        }, f, indent=2)

if __name__ == "__main__":
    analyze_intent_distribution()
