import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import GOLDEN_SET_DIR, RESULTS_DIR
from src.agent import CustomerSupportAgent
from src.evaluation import Evaluator, compute_human_llm_agreement

GOLDEN_SET_PATH = GOLDEN_SET_DIR / "golden_set.json"
EVAL_RESULTS_PATH = RESULTS_DIR / "evaluation_summary.json"

def main():
    print(f"Loading Audited Golden Evaluation Set from {GOLDEN_SET_PATH}...")
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    evaluator = Evaluator(use_llm_judge=True)

    print(f"\n==================================================")
    print(f"       RUNNING SYSTEM EVALUATION SUITE            ")
    print(f"==================================================")

    # 1. Evaluate Baseline 1: Trivial Majority Agent
    print("\n[1/3] Evaluating Baseline 1: Trivial Majority Agent (Auto-handle Everything)...")
    agent_b1 = CustomerSupportAgent(mode="trivial_baseline")
    res_b1 = evaluator.evaluate_golden_set(agent_b1, golden_set)

    # 2. Evaluate Baseline 2: TF-IDF Intent + Safety Escalation + Non-RAG Static Canned Templates
    print("\n[2/3] Evaluating Baseline 2: TF-IDF + Policy + Non-RAG Static Technical Templates...")
    agent_b2 = CustomerSupportAgent(mode="baseline_2")
    res_b2 = evaluator.evaluate_golden_set(agent_b2, golden_set)

    # 3. Evaluate Main System: Full Grounded RAG Agent Pipeline
    print("\n[3/3] Evaluating Main System: Full Grounded RAG Agent Pipeline...")
    agent_main = CustomerSupportAgent(mode="main")
    res_main = evaluator.evaluate_golden_set(agent_main, golden_set)

    # 4. Human vs. LLM-as-Judge Agreement Study
    print("\n[4/4] Running Human vs. LLM-as-Judge Statistical Agreement Study...")
    agreement_metrics = compute_human_llm_agreement(res_main["detailed_results"])

    summary = {
        "baseline_1_trivial": res_b1["metrics"],
        "baseline_2_tfidf_template": res_b2["metrics"],
        "main_system": res_main["metrics"],
        "human_llm_judge_agreement": agreement_metrics
    }

    print("\n" + "=" * 78)
    print("                EVALUATION COMPARISON BENCHMARK RESULTS           ")
    print("=" * 78)
    print(f"{'Metric':<34} | {'Baseline 1':<12} | {'Baseline 2':<12} | {'Main System':<12}")
    print("-" * 78)
    print(f"{'Intent Accuracy':<34} | {res_b1['metrics']['intent_classification']['accuracy']:<12.4f} | {res_b2['metrics']['intent_classification']['accuracy']:<12.4f} | {res_main['metrics']['intent_classification']['accuracy']:<12.4f}")
    print(f"{'Intent Macro F1':<34} | {res_b1['metrics']['intent_classification']['macro_f1']:<12.4f} | {res_b2['metrics']['intent_classification']['macro_f1']:<12.4f} | {res_main['metrics']['intent_classification']['macro_f1']:<12.4f}")
    print(f"{'Auto-Handle Rate':<34} | {res_b1['metrics']['escalation_policy']['auto_handle_rate']:<12.4f} | {res_b2['metrics']['escalation_policy']['auto_handle_rate']:<12.4f} | {res_main['metrics']['escalation_policy']['auto_handle_rate']:<12.4f}")
    print(f"{'Escalation Rate':<34} | {res_b1['metrics']['escalation_policy']['escalation_rate']:<12.4f} | {res_b2['metrics']['escalation_policy']['escalation_rate']:<12.4f} | {res_main['metrics']['escalation_policy']['escalation_rate']:<12.4f}")
    print(f"{'Escalation Precision':<34} | {res_b1['metrics']['escalation_policy']['escalation_precision']:<12.4f} | {res_b2['metrics']['escalation_policy']['escalation_precision']:<12.4f} | {res_main['metrics']['escalation_policy']['escalation_precision']:<12.4f}")
    print(f"{'Escalation Recall':<34} | {res_b1['metrics']['escalation_policy']['escalation_recall']:<12.4f} | {res_b2['metrics']['escalation_policy']['escalation_recall']:<12.4f} | {res_main['metrics']['escalation_policy']['escalation_recall']:<12.4f}")
    print(f"{'Dangerous Auto-Handle Count':<34} | {res_b1['metrics']['escalation_policy']['dangerous_auto_handle_count']:<12} | {res_b2['metrics']['escalation_policy']['dangerous_auto_handle_count']:<12} | {res_main['metrics']['escalation_policy']['dangerous_auto_handle_count']:<12}")
    print(f"{'Dangerous Auto-Handle Rate':<34} | {res_b1['metrics']['escalation_policy']['dangerous_auto_handle_rate']:<12.4f} | {res_b2['metrics']['escalation_policy']['dangerous_auto_handle_rate']:<12.4f} | {res_main['metrics']['escalation_policy']['dangerous_auto_handle_rate']:<12.4f}")
    print(f"{'Reply Groundedness (1-10)':<34} | {res_b1['metrics']['reply_quality']['avg_groundedness']:<12.2f} | {res_b2['metrics']['reply_quality']['avg_groundedness']:<12.2f} | {res_main['metrics']['reply_quality']['avg_groundedness']:<12.2f}")
    print(f"{'Intent Alignment (1-10)':<34} | {res_b1['metrics']['reply_quality']['avg_intent_alignment']:<12.2f} | {res_b2['metrics']['reply_quality']['avg_intent_alignment']:<12.2f} | {res_main['metrics']['reply_quality']['avg_intent_alignment']:<12.2f}")
    print(f"{'Tone & Safety (1-10)':<34} | {res_b1['metrics']['reply_quality']['avg_tone_and_safety']:<12.2f} | {res_b2['metrics']['reply_quality']['avg_tone_and_safety']:<12.2f} | {res_main['metrics']['reply_quality']['avg_tone_and_safety']:<12.2f}")
    print(f"{'Overall Quality (1-10)':<34} | {res_b1['metrics']['reply_quality']['avg_overall_quality']:<12.2f} | {res_b2['metrics']['reply_quality']['avg_overall_quality']:<12.2f} | {res_main['metrics']['reply_quality']['avg_overall_quality']:<12.2f}")
    print("=" * 78)

    print("\n--- HUMAN VS. LLM-AS-JUDGE AGREEMENT STUDY ---")
    print(f"Sample Size               : {agreement_metrics['sample_size']} examples")
    print(f"Mean Absolute Error (MAE) : {agreement_metrics['mean_absolute_error_mae']}")
    print(f"Pearson Correlation (r)   : {agreement_metrics['pearson_correlation_r']}")
    print(f"1-Point Agreement Rate    : {agreement_metrics['one_point_agreement_rate']:.2%}")

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
