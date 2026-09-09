from typing import Dict, Any, List
from src.preprocessing import clean_text
from src.classifier import TFIDFIntentClassifier, MajorityIntentClassifier
from src.retriever import HistoricalRetriever
from src.escalation import EscalationPolicy
from src.generator import GroundedReplyGenerator

class CustomerSupportAgent:
    """
    End-to-End Grounded Customer Support Agent for AppleSupport.
    Integrates Preprocessing, Intent Classification, Retrieval, Escalation, and Reply Generation.
    """
    def __init__(self, top_k: int = 3, mode: str = "main"):
        self.mode = mode
        self.top_k = top_k
        self.classifier = TFIDFIntentClassifier() if mode in ["main", "baseline_2"] else MajorityIntentClassifier()
        self.retriever = HistoricalRetriever(top_k=top_k)
        self.escalation_policy = EscalationPolicy()
        self.generator = GroundedReplyGenerator()


    def process_message(self, message: str, tweet_id: int = None) -> Dict[str, Any]:
        """
        Processes an incoming customer tweet and produces structured decisions & response.
        """
        # Step 1: Preprocessing
        cleaned_msg = clean_text(message)

        # Step 2: Intent Classification
        predicted_intent, confidence = self.classifier.predict(cleaned_msg)

        # Step 3: Historical Similar-Case Retrieval
        retrieved_cases = []
        if self.mode == "main":
            retrieved_cases = self.retriever.retrieve(cleaned_msg, top_k=self.top_k, exclude_tweet_id=tweet_id)

        # Step 4: Escalation Policy Evaluation
        if self.mode == "trivial_baseline":
            decision = "AUTO_HANDLE"
            reason = "Trivial Baseline: Auto-handle everything with canned reply."
        else:
            decision, reason = self.escalation_policy.evaluate(
                predicted_intent=predicted_intent,
                confidence=confidence,
                retrieved_cases=retrieved_cases
            )

        # Step 5: Grounded Reply Generation
        if self.mode == "trivial_baseline":
            reply = "Thank you for reaching out to Apple Support. Please restart your device or visit https://support.apple.com for assistance."
        elif self.mode == "baseline_2":
            reply = self.generator.generate_reply(
                customer_message=cleaned_msg,
                predicted_intent=predicted_intent,
                retrieved_cases=retrieved_cases,
                decision=decision,
                reason=reason,
                use_rag=False
            )
        else:
            reply = self.generator.generate_reply(
                customer_message=cleaned_msg,
                predicted_intent=predicted_intent,
                retrieved_cases=retrieved_cases,
                decision=decision,
                reason=reason,
                use_rag=True
            )

        return {
            "customer_message": cleaned_msg,
            "intent": predicted_intent,
            "intent_confidence": round(confidence, 4),
            "decision": decision,
            "reason": reason,
            "reply": reply,
            "evidence": retrieved_cases
        }

