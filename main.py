import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.agent import CustomerSupportAgent

def run_demo():
    print("=" * 70)
    print("           @AppleSupport AI Support Agent - Demonstration         ")
    print("=" * 70)

    agent = CustomerSupportAgent(mode="main")

    sample_queries = [
        ("Query 1 (Software Update)", "Downloaded iOS 11.2 on my iPhone 8 Plus but my apps keep freezing when opening."),
        ("Query 2 (Account Security)", "My Apple ID has been locked for security reasons and I can't sign into iCloud!"),
        ("Query 3 (Battery Drain)", "My battery is draining 20% every hour since yesterday. How do I fix this?"),
        ("Query 4 (Hardware Repair)", "Screen touch is completely broken after dropping my phone. How much is screen replacement?")
    ]

    for label, msg in sample_queries:
        print(f"\n--- {label} ---")
        print(f"Customer Message: \"{msg}\"")
        output = agent.process_message(msg)
        print(f"Predicted Intent : {output['intent']} (Confidence: {output['intent_confidence']:.2%})")
        print(f"Handling Decision: {output['decision']}")
        print(f"Decision Reason  : {output['reason']}")
        print(f"Agent Response   : \"{output['reply']}\"")
        if output['evidence']:
            top_ev = output['evidence'][0]
            print(f"Top RAG Evidence : Match score {top_ev['similarity_score']:.2f} | \"{top_ev['customer_message']}\"")
        print("-" * 70)

def run_interactive():
    agent = CustomerSupportAgent(mode="main")
    print("\n" + "=" * 70)
    print("      INTERACTIVE @AppleSupport AGENT CLI (Type 'exit' to quit)     ")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter customer message > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive CLI. Goodbye!")
                break

            output = agent.process_message(user_input)
            print("\n[AGENT OUTPUT]")
            print(f"Intent            : {output['intent']} (Confidence: {output['intent_confidence']:.2%})")
            print(f"Handling Decision : {output['decision']}")
            print(f"Decision Reason   : {output['reason']}")
            print(f"Generated Reply   : {output['reply']}")
            if output['evidence']:
                print(f"Top Retrieved Case: {output['evidence'][0]['brand_response']}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive CLI.")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_demo()
        print("\nTip: Run 'python main.py --interactive' to test custom tweets!")
