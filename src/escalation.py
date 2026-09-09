from typing import Tuple, List, Dict, Any

class EscalationPolicy:
    """
    Explicit Multi-Signal Safety & Escalation Policy for AppleSupport Agent.
    Prioritizes safety over aggressive automation.
    """
    def __init__(self, confidence_threshold: float = 0.35, similarity_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

        # High-risk intent categories that mandatory require human touch or private DM authorization
        self.sensitive_intents = {
            "ACCOUNT_APPLE_ID_SECURITY": "Account security & Apple ID lock issues require private authentication.",
            "BILLING_APP_STORE_SUBSCRIPTION": "Billing, refunds, and payment transactions require user account verification.",
            "HARDWARE_DISPLAY_AUDIO": "Physical hardware damage (display/audio) requires physical Genius Bar inspection.",
            "HARDWARE_REPAIR_SERVICE_INQUIRY": "Repair appointments and service cost quotes require official store agent booking.",
            "OTHER_AMBIGUOUS_COMPLAINT": "Query is ambiguous or lacks technical detail required for automated resolution."
        }

    def evaluate(
        self,
        predicted_intent: str,
        confidence: float,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Evaluates query signals and decides AUTO_HANDLE vs ESCALATE.
        Returns:
            decision: "AUTO_HANDLE" or "ESCALATE"
            reason: Explanation string for decision
        """
        # Signal 1: Sensitive / Mandatory Escalation Category
        if predicted_intent in self.sensitive_intents:
            return "ESCALATE", f"Mandatory Escalation: {self.sensitive_intents[predicted_intent]}"

        # Signal 2: Low Classifier Confidence
        if confidence < self.confidence_threshold:
            return "ESCALATE", f"Low Intent Confidence ({confidence:.2f} < {self.confidence_threshold}). Escalated to prevent misrouting."

        # Signal 3: Insufficient Historical Evidence (Weak Retrieval Similarity)
        if not retrieved_cases:
            return "ESCALATE", "Insufficient Evidence: No relevant historical cases found in support corpus."

        max_sim = max([case.get('similarity_score', 0.0) for case in retrieved_cases])
        if max_sim < self.similarity_threshold:
            return "ESCALATE", f"Weak Retrieval Match: Top historical case similarity ({max_sim:.2f}) is below safety threshold ({self.similarity_threshold})."

        # Signal 4: Safe Autonomous Resolution
        return "AUTO_HANDLE", f"Safe Auto-Handle: Intent '{predicted_intent}' identified with high confidence ({confidence:.2f}) and grounded by historical case match ({max_sim:.2f})."
