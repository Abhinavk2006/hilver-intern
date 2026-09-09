import os
import re
from typing import List, Dict, Any

class GroundedReplyGenerator:
    """
    Generates grounded customer support replies using historical retrieval evidence.
    Supports both LLM API generation (Google Gemini) and robust template-fallback.
    """
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.client = None
        
        # Check if Gemini API key is available in environment
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if use_llm and api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
                print("Initialized Google Gemini client for response generation.")
            except Exception as e:
                print(f"Could not initialize Gemini client: {e}. Using grounded template generator.")

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_cases: List[Dict[str, Any]],
        decision: str,
        reason: str
    ) -> str:
        """
        Generates a grounded, brand-consistent reply.
        """
        top_case = retrieved_cases[0] if retrieved_cases else None
        evidence_reply = top_case['brand_response'] if top_case else ""

        # If escalation is decided, generate a polite escalation response directing to DM or human support
        if decision == "ESCALATE":
            if "ACCOUNT" in predicted_intent or "BILLING" in predicted_intent:
                return "We want to help resolve this for you safely. Because this involves personal account details, please send us a Direct Message (DM) so we can securely assist: https://t.co/AppleSupportDM"
            elif "HARDWARE" in predicted_intent:
                return "To assist with your device hardware, we recommend visiting your local Apple Store or making a Genius Bar appointment: https://support.apple.com/repair"
            else:
                return "We understand your issue and want to make sure it's handled properly. Please Send us a Direct Message with your device model and iOS version so our team can help: https://t.co/AppleSupportDM"

        # If LLM API is available and enabled
        if self.client:
            try:
                prompt = f"""You are an official Apple Support agent on Twitter (@AppleSupport).
Customer Message: "{customer_message}"
Predicted Category: {predicted_intent}

Historical AppleSupport Evidence Cases:
{chr(10).join([f"- Query: {c['customer_message']} -> Response: {c['brand_response']}" for c in retrieved_cases[:2]])}

Instructions:
- Write a concise (under 280 chars), polite, and customer-friendly reply grounded strictly in the historical evidence.
- Never invent policies, refunds, or hardware guarantees not present in evidence.
- Provide standard Apple support steps (e.g. restart, update, check settings) or link guidance.
"""
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                pass  # Fallback to evidence template

        # Grounded Evidence Template Generator:
        # Extract support links or core guidance from historical evidence
        urls = re.findall(r'https?://\S+', evidence_reply)
        link_str = f" Learn more here: {urls[0]}" if urls else ""

        if predicted_intent == "SOFTWARE_UPDATE_ISSUES":
            return f"We'd be happy to look into this update issue with you. First, make sure your device has a recent backup, then check for updates in Settings > General > Software Update.{link_str}"
        elif predicted_intent == "BATTERY_POWER_ISSUES":
            return f"We know battery life is important. Check your battery usage breakdown in Settings > Battery to see which apps are consuming power.{link_str}"
        elif predicted_intent == "DEVICE_FREEZE_REBOOT":
            return f"Let's get your device back to working smoothly. Try force restarting your device following the steps here: https://support.apple.com/HT201412"
        elif predicted_intent == "ICLOUD_STORAGE_SYNC":
            return f"We can help with your storage. Check your current storage usage in Settings > [Your Name] > iCloud > Manage Storage.{link_str}"
        elif predicted_intent == "CONNECTIVITY_NETWORK_BLUETOOTH":
            return f"Let's get you connected again. Try resetting Network Settings in Settings > General > Reset > Reset Network Settings.{link_str}"
        elif predicted_intent == "GENERAL_HOW_TO_INFO":
            if evidence_reply:
                return f"Here to help! {evidence_reply[:140]}..."
            return f"You can adjust and manage this setting directly in Settings. Let us know if you need step-by-step guidance!"

        # Default grounded response using top evidence case pattern
        if evidence_reply:
            return f"We'd be glad to help with this! Follow these steps from Apple Support: {evidence_reply[:150]}"
        return "We'd like to help you with this. Please check your settings or restart your device to see if the issue persists."
