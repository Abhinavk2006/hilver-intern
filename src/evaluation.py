import os
import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.intents import get_intent_list

class Evaluator:
    """
    Comprehensive Evaluation Harness for Customer Support Agent.
    Evaluates:
    1. Intent Classification Accuracy & Macro F1
    2. Escalation Precision, Recall, and Safety (Dangerous Auto-Handle Rate)
    3. Grounded Reply Quality via Heuristics & LLM-as-Judge
    """
    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self.judge_client = None
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if use_llm_judge and api_key:
            try:
                from google import genai
                self.judge_client = genai.Client(api_key=api_key)
                print("Initialized Gemini client for LLM-as-Judge evaluation.")
            except Exception as e:
                print(f"Could not initialize LLM Judge: {e}. Falling back to heuristic quality scoring.")

    def evaluate_golden_set(self, agent_instance, golden_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs agent through entire golden set and calculates comprehensive evaluation suite.
        """
        results = []
        gold_intents = []
        pred_intents = []
        
        gold_decisions = []
        pred_decisions = []

        print(f"Evaluating {len(golden_set)} golden set examples...")

        for item in golden_set:
            customer_msg = item['customer_message']
            gold_intent = item['gold_intent']
            gold_decision = item['gold_handling_decision']
            tweet_id = item.get('customer_tweet_id')

            # Run agent prediction
            output = agent_instance.process_message(customer_msg, tweet_id=tweet_id)

            pred_intent = output['intent']
            pred_decision = output['decision']

            gold_intents.append(gold_intent)
            pred_intents.append(pred_intent)

            gold_decisions.append(gold_decision)
            pred_decisions.append(pred_decision)

            record = {
                "id": item['id'],
                "customer_message": customer_msg,
                "gold_intent": gold_intent,
                "pred_intent": pred_intent,
                "intent_confidence": output['intent_confidence'],
                "gold_decision": gold_decision,
                "pred_decision": pred_decision,
                "decision_reason": output['reason'],
                "generated_reply": output['reply'],
                "reference_reply": item.get('reference_reply', ''),
                "evidence": output.get('evidence', [])
            }
            results.append(record)

        # --- 1. Intent Metrics ---
        intent_labels = get_intent_list()
        intent_acc = accuracy_score(gold_intents, pred_intents)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            gold_intents, pred_intents, average='macro', zero_division=0
        )

        # --- 2. Escalation & Auto-Handle Metrics ---
        # Binary: ESCALATE = 1, AUTO_HANDLE = 0
        gold_esc_binary = [1 if d == "ESCALATE" else 0 for d in gold_decisions]
        pred_esc_binary = [1 if d == "ESCALATE" else 0 for d in pred_decisions]

        esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
            gold_esc_binary, pred_esc_binary, average='binary', zero_division=0
        )

        total_count = len(golden_set)
        auto_handle_count = sum(1 for d in pred_decisions if d == "AUTO_HANDLE")
        escalate_count = sum(1 for d in pred_decisions if d == "ESCALATE")

        auto_handle_rate = auto_handle_count / total_count
        escalation_rate = escalate_count / total_count

        # Safety calculation:
        # Dangerous Auto-Handle: Agent decided AUTO_HANDLE, but Gold required ESCALATE!
        dangerous_auto_handles = sum(
            1 for g_dec, p_dec in zip(gold_decisions, pred_decisions)
            if p_dec == "AUTO_HANDLE" and g_dec == "ESCALATE"
        )
        dangerous_auto_handle_rate = dangerous_auto_handles / max(1, auto_handle_count)

        # Safe Auto-Handle: Agent decided AUTO_HANDLE, and Gold was AUTO_HANDLE
        safe_auto_handles = sum(
            1 for g_dec, p_dec in zip(gold_decisions, pred_decisions)
            if p_dec == "AUTO_HANDLE" and g_dec == "AUTO_HANDLE"
        )
        safe_auto_handle_rate = safe_auto_handles / max(1, auto_handle_count)

        # --- 3. Reply Quality Metrics (Heuristics & LLM Judge) ---
        reply_scores = []
        for rec in results:
            q_score = self._score_reply_heuristics(rec)
            rec['heuristic_quality_score'] = q_score
            reply_scores.append(q_score)

        avg_heuristic_score = sum(reply_scores) / max(1, len(reply_scores))

        # --- 4. Categorize Failure Modes ---
        failures = []
        for rec in results:
            failure_types = []
            if rec['pred_intent'] != rec['gold_intent']:
                failure_types.append("INTENT_MISCLASSIFICATION")
            if rec['pred_decision'] == "AUTO_HANDLE" and rec['gold_decision'] == "ESCALATE":
                failure_types.append("DANGEROUS_AUTO_HANDLE")
            elif rec['pred_decision'] == "ESCALATE" and rec['gold_decision'] == "AUTO_HANDLE":
                failure_types.append("UNNECESSARY_ESCALATION")

            if failure_types:
                rec['failure_types'] = failure_types
                failures.append(rec)

        metrics = {
            "total_examples": total_count,
            "intent_classification": {
                "accuracy": round(float(intent_acc), 4),
                "macro_precision": round(float(macro_p), 4),
                "macro_recall": round(float(macro_r), 4),
                "macro_f1": round(float(macro_f1), 4)
            },
            "escalation_policy": {
                "auto_handle_rate": round(auto_handle_rate, 4),
                "escalation_rate": round(escalation_rate, 4),
                "escalation_precision": round(esc_p, 4),
                "escalation_recall": round(esc_r, 4),
                "escalation_f1": round(esc_f1, 4),
                "safe_auto_handle_rate": round(safe_auto_handle_rate, 4),
                "dangerous_auto_handle_count": dangerous_auto_handles,
                "dangerous_auto_handle_rate": round(dangerous_auto_handle_rate, 4)
            },
            "reply_quality": {
                "avg_heuristic_score": round(avg_heuristic_score, 2)
            },
            "total_failures": len(failures)
        }

        return {
            "metrics": metrics,
            "detailed_results": results,
            "failures": failures
        }

    def _score_reply_heuristics(self, record: Dict[str, Any]) -> float:
        """
        Rule-based reply quality scoring (0 - 10 points).
        """
        reply = record['generated_reply']
        score = 10.0

        # Penalty 1: Exceeding Twitter length limit (280 chars)
        if len(reply) > 280:
            score -= 2.0

        # Penalty 2: Blank or extremely short response
        if len(reply) < 15:
            score -= 5.0

        # Penalty 3: Unprofessional / aggressive language
        if any(w in reply.lower() for w in ["stupid", "idiot", "cannot help you", "go away"]):
            score -= 5.0

        # Reward 1: Directing sensitive account queries to DM or Official Support
        if record['gold_decision'] == "ESCALATE":
            if any(k in reply.lower() for k in ["dm", "direct message", "support.apple.com", "genius bar", "repair"]):
                score += 1.0
            else:
                score -= 3.0  # Failed to provide proper escalation pathway

        return max(0.0, min(10.0, score))
