import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import GOLDEN_SET_DIR, RESULTS_DIR
from src.agent import CustomerSupportAgent
from src.evaluation import Evaluator

GOLDEN_SET_PATH = GOLDEN_SET_DIR / "golden_set.json"
EVAL_RESULTS_PATH = RESULTS_DIR / "evaluation_summary.json"

def main():
    print(f"Loading Golden Evaluation Set from {GOLDEN_SET_PATH}...")
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    evaluator = Evaluator(use_llm_judge=False)

    print(f"\n==================================================")
    print(f"       RUNNING SYSTEM EVALUATION SUITE            ")
    print(f"==================================================")

    # 1. Evaluate Baseline 1: Trivial Majority Agent
    print("\n[1/3] Evaluating Baseline 1: Trivial Majority Agent (Auto-handle Everything)...")
    agent_b1 = CustomerSupportAgent(mode="trivial_baseline")
    res_b1 = evaluator.evaluate_golden_set(agent_b1, golden_set)

    # 2. Evaluate Baseline 2: TF-IDF Intent + Escalation + Template Generator (No LLM)
    print("\n[2/3] Evaluating Baseline 2: TF-IDF + Escalation + Grounded Templates...")
    agent_b2 = CustomerSupportAgent(mode="main")
    agent_b2.generator.use_llm = False
    agent_b2.generator.client = None
    res_b2 = evaluator.evaluate_golden_set(agent_b2, golden_set)

    # 3. Evaluate Main System: Full Pipeline
    print("\n[3/3] Evaluating Main System: Full Grounded Agent Pipeline...")
    agent_main = CustomerSupportAgent(mode="main")
    res_main = evaluator.evaluate_golden_set(agent_main, golden_set)

    summary = {
        "baseline_1_trivial": res_b1["metrics"],
        "baseline_2_tfidf_template": res_b2["metrics"],
        "main_system": res_main["metrics"]
    }

    print("\n" + "=" * 60)
    print("                EVALUATION COMPARISON RESULTS           ")
    print("=" * 60)
    print(f"{'Metric':<32} | {'Baseline 1':<12} | {'Baseline 2':<12} | {'Main System':<12}")
    print("-" * 75)
    print(f"{'Intent Accuracy':<32} | {res_b1['metrics']['intent_classification']['accuracy']:<12.4f} | {res_b2['metrics']['intent_classification']['accuracy']:<12.4f} | {res_main['metrics']['intent_classification']['accuracy']:<12.4f}")
    print(f"{'Intent Macro F1':<32} | {res_b1['metrics']['intent_classification']['macro_f1']:<12.4f} | {res_b2['metrics']['intent_classification']['macro_f1']:<12.4f} | {res_main['metrics']['intent_classification']['macro_f1']:<12.4f}")
    print(f"{'Auto-Handle Rate':<32} | {res_b1['metrics']['escalation_policy']['auto_handle_rate']:<12.4f} | {res_b2['metrics']['escalation_policy']['auto_handle_rate']:<12.4f} | {res_main['metrics']['escalation_policy']['auto_handle_rate']:<12.4f}")
    print(f"{'Escalation Rate':<32} | {res_b1['metrics']['escalation_policy']['escalation_rate']:<12.4f} | {res_b2['metrics']['escalation_policy']['escalation_rate']:<12.4f} | {res_main['metrics']['escalation_policy']['escalation_rate']:<12.4f}")
    print(f"{'Escalation Precision':<32} | {res_b1['metrics']['escalation_policy']['escalation_precision']:<12.4f} | {res_b2['metrics']['escalation_policy']['escalation_precision']:<12.4f} | {res_main['metrics']['escalation_policy']['escalation_precision']:<12.4f}")
    print(f"{'Escalation Recall':<32} | {res_b1['metrics']['escalation_policy']['escalation_recall']:<12.4f} | {res_b2['metrics']['escalation_policy']['escalation_recall']:<12.4f} | {res_main['metrics']['escalation_policy']['escalation_recall']:<12.4f}")
    print(f"{'Dangerous Auto-Handle Count':<32} | {res_b1['metrics']['escalation_policy']['dangerous_auto_handle_count']:<12} | {res_b2['metrics']['escalation_policy']['dangerous_auto_handle_count']:<12} | {res_main['metrics']['escalation_policy']['dangerous_auto_handle_count']:<12}")
    print(f"{'Dangerous Auto-Handle Rate':<32} | {res_b1['metrics']['escalation_policy']['dangerous_auto_handle_rate']:<12.4f} | {res_b2['metrics']['escalation_policy']['dangerous_auto_handle_rate']:<12.4f} | {res_main['metrics']['escalation_policy']['dangerous_auto_handle_rate']:<12.4f}")
    print(f"{'Avg Reply Quality (0-10)':<32} | {res_b1['metrics']['reply_quality']['avg_heuristic_score']:<12.2f} | {res_b2['metrics']['reply_quality']['avg_heuristic_score']:<12.2f} | {res_main['metrics']['reply_quality']['avg_heuristic_score']:<12.2f}")
    print("=" * 60)

    # Save detailed evaluation artifacts
    full_output = {
        "summary": summary,
        "main_system_failures": res_main["failures"],
        "main_system_detailed_results": res_main["detailed_results"]
    }

    with open(EVAL_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[SUCCESS] Evaluation report saved to {EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    main()
