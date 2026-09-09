"""
Intent Taxonomy for AppleSupport Customer Agent.
Derived empirically from 104,554 AppleSupport customer conversations.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class IntentDefinition:
    name: str
    definition: str
    inclusion_criteria: str
    exclusion_criteria: str
    examples: List[str]
    default_handling: str  # AUTO_HANDLE or ESCALATE
    keywords: List[str]

INTENT_TAXONOMY: Dict[str, IntentDefinition] = {
    "SOFTWARE_UPDATE_ISSUES": IntentDefinition(
        name="SOFTWARE_UPDATE_ISSUES",
        definition="Issues related to downloading, installing, or post-update glitches of iOS/watchOS/macOS.",
        inclusion_criteria="Mentions of software update, iOS version, installation stuck, or bugs occurring after updating.",
        exclusion_criteria="Hardware damage, battery drain specifically, account login issues.",
        examples=[
            "Downloaded 11.2 on iPhone 8 Plus but no Apple Cash option in settings?",
            "I updated my iPhone to iOS 11 and now my apps crash constantly.",
            "My update is stuck on preparing update for 3 hours."
        ],
        default_handling="AUTO_HANDLE",
        keywords=["update", "ios", "upgrade", "version", "install", "download", "beta"]
    ),
    "BATTERY_POWER_ISSUES": IntentDefinition(
        name="BATTERY_POWER_ISSUES",
        definition="Abnormal battery drain, charging failure, device overheating, or battery health drops.",
        inclusion_criteria="Mentions of battery life, fast drain, overheating, charger not working, or phone dying quickly.",
        exclusion_criteria="Screen damage, software update installation errors.",
        examples=[
            "After a few days with the new update my battery seems to be draining a lot faster",
            "My iPhone won't charge past 80% and gets super hot.",
            "Battery health dropped to 75% overnight."
        ],
        default_handling="AUTO_HANDLE",
        keywords=["battery", "drain", "charge", "charging", "overheat", "dying", "power"]
    ),
    "DEVICE_FREEZE_REBOOT": IntentDefinition(
        name="DEVICE_FREEZE_REBOOT",
        definition="Critical device instability: boot loops, continuous random restarts, stuck on Apple logo, or unresponsive screen.",
        inclusion_criteria="Mentions of phone restarting by itself, stuck on boot logo, black screen, or total unresponsiveness.",
        exclusion_criteria="Minor app crash without system reboot.",
        examples=[
            "My phone keeps restarting by itself every 2 minutes SOS",
            "Screen is completely frozen and hard reset doesn't work.",
            "Stuck on Apple logo after restart."
        ],
        default_handling="AUTO_HANDLE",
        keywords=["restart", "reboot", "freeze", "frozen", "stuck", "loop", "boot", "black screen"]
    ),
    "ACCOUNT_APPLE_ID_SECURITY": IntentDefinition(
        name="ACCOUNT_APPLE_ID_SECURITY",
        definition="Apple ID access, account lockouts, passcode resets, 2FA verification, or disabled accounts.",
        inclusion_criteria="Mentions of locked Apple ID, forgotten password, passcode disabled, 2-factor code not received.",
        exclusion_criteria="App Store payment decline without account lockout.",
        examples=[
            "My Apple ID has been disabled for security reasons, how do I unlock it?",
            "Forgot passcode and iPhone is disabled connect to iTunes.",
            "Not receiving 2FA verification code on my trusted device."
        ],
        default_handling="ESCALATE",
        keywords=["apple id", "password", "lock", "locked", "passcode", "disabled", "sign in", "login", "2fa"]
    ),
    "ICLOUD_STORAGE_SYNC": IntentDefinition(
        name="ICLOUD_STORAGE_SYNC",
        definition="iCloud storage limits, backup creation failures, photo library sync, or file recovery.",
        inclusion_criteria="Mentions of iCloud, storage full, photo sync, backup failed, drive sync.",
        exclusion_criteria="Physical device internal storage expansion.",
        examples=[
            "iCloud backup says last backup could not be completed.",
            "Bought 50GB iCloud storage but phone still says storage full.",
            "Photos are not syncing between Mac and iPhone."
        ],
        default_handling="AUTO_HANDLE",
        keywords=["icloud", "storage", "backup", "sync", "photos", "cloud"]
    ),
    "BILLING_APP_STORE_SUBSCRIPTION": IntentDefinition(
        name="BILLING_APP_STORE_SUBSCRIPTION",
        definition="Unauthorized charges, App Store refunds, subscription management, Apple Pay declined payments.",
        inclusion_criteria="Mentions of charges, invoice, refund, subscription cancellation, card decline, App Store billing.",
        exclusion_criteria="Free app download technical errors.",
        examples=[
            "I was charged $9.99 for a subscription I cancelled last month.",
            "How do I request a refund for an accidental App Store purchase?",
            "Apple Pay keeps declining my debit card."
        ],
        default_handling="ESCALATE",
        keywords=["billing", "charge", "refund", "subscription", "app store", "purchase", "apple pay", "invoice"]
    ),
    "CONNECTIVITY_NETWORK_BLUETOOTH": IntentDefinition(
        name="CONNECTIVITY_NETWORK_BLUETOOTH",
        definition="Wireless connectivity problems involving Wi-Fi, Bluetooth, cellular network, or AirDrop.",
        inclusion_criteria="Mentions of Wi-Fi dropping, Bluetooth pairing failure, no service / searching, AirDrop missing.",
        exclusion_criteria="Device total freeze or boot loop.",
        examples=[
            "Wi-Fi keeps disconnecting every time phone locks.",
            "AirPods won't pair via Bluetooth to my iPhone.",
            "Cellular signal says No Service after traveling."
        ],
        default_handling="AUTO_HANDLE",
        keywords=["wifi", "wi-fi", "bluetooth", "cellular", "signal", "service", "pair", "airdrop", "connection"]
    ),
    "HARDWARE_DISPLAY_AUDIO": IntentDefinition(
        name="HARDWARE_DISPLAY_AUDIO",
        definition="Physical or component issues with display screen, touch response, speakers, microphone, or camera.",
        inclusion_criteria="Mentions of cracked display, touch screen unresponsive, speaker quiet/distorted, mic not picking up voice.",
        exclusion_criteria="System software boot loops without hardware physical damage.",
        examples=[
            "Screen touch sensitivity isn't working on the left edge.",
            "Speaker sound is extremely muffled during phone calls.",
            "Camera screen turns completely black when opening app."
        ],
        default_handling="ESCALATE",
        keywords=["screen", "display", "touch", "speaker", "audio", "microphone", "sound", "camera"]
    ),
    "HARDWARE_REPAIR_SERVICE_INQUIRY": IntentDefinition(
        name="HARDWARE_REPAIR_SERVICE_INQUIRY",
        definition="Inquiries about Genius Bar appointments, repair pricing, AppleCare coverage, or device mail-in service.",
        inclusion_criteria="Mentions of Genius Bar, appointment, repair cost, warranty, AppleCare, replacement.",
        exclusion_criteria="General software troubleshooting without repair intent.",
        examples=[
            "How much does it cost to replace iPhone screen without AppleCare?",
            "Need to book a Genius Bar appointment at local Apple Store.",
            "How long does mail-in repair service take?"
        ],
        default_handling="ESCALATE",
        keywords=["repair", "genius", "appointment", "store", "warranty", "applecare", "replacement", "cost"]
    ),
    "GENERAL_HOW_TO_INFO": IntentDefinition(
        name="GENERAL_HOW_TO_INFO",
        definition="Standard guidance, setting configurations, feature explanations, or usage instructions.",
        inclusion_criteria="Questions starting with how to, where is setting, how do I turn on feature.",
        exclusion_criteria="Broken features or device errors.",
        examples=[
            "How do I enable Dark Mode on iOS 11?",
            "Where do I find screen recording in Control Center?",
            "How to transfer data from old iPhone to new iPhone?"
        ],
        default_handling="AUTO_HANDLE",
        keywords=["how to", "how do i", "setting", "settings", "feature", "where is"]
    ),
    "OTHER_AMBIGUOUS_COMPLAINT": IntentDefinition(
        name="OTHER_AMBIGUOUS_COMPLAINT",
        definition="Vague complaints, multi-topic queries, ranting, or messages missing specific actionable context.",
        inclusion_criteria="Rants without clear error symptoms, general complaints, or ambiguous short queries.",
        exclusion_criteria="Queries matching one of the specific 10 technical categories above.",
        examples=[
            "What the heck is wrong with Apple products lately?!",
            "Fix this now I hate this phone",
            "Hello Apple support please help"
        ],
        default_handling="ESCALATE",
        keywords=[]
    )
}

def get_intent_list() -> List[str]:
    return list(INTENT_TAXONOMY.keys())

def get_intent_meta(intent_name: str) -> IntentDefinition:
    return INTENT_TAXONOMY.get(intent_name, INTENT_TAXONOMY["OTHER_AMBIGUOUS_COMPLAINT"])
