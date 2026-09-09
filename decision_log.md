# Architectural Decision Log: AppleSupport AI Customer Support Agent

This log documents key engineering and design decisions made during the implementation of the production-minded AI customer-support agent for `@AppleSupport`.

---

### Decision 1: Brand Selection — AppleSupport over Uber/Delta
- **Decision**: Select `@AppleSupport` as the single brand for building the AI support agent.
- **Why Made**: `@AppleSupport` features the highest volume in the dataset (>213,000 dialogue turns) with clear distinctions between public technical troubleshooting (iOS updates, settings) and sensitive private inquiries (Apple ID locks, billing refunds, hardware repairs).
- **Alternative Considered**: `@Uber_Support` or `@Delta`.
- **Why Rejected**: Uber and Delta involve heavy real-time trip/flight PII and geolocation-dependent context, whereas Apple Support provides a well-structured technical domain hierarchy suitable for grounded RAG and clear escalation boundaries.

---

### Decision 2: Strict Text Preprocessing & Handle Normalization
- **Decision**: Implement a custom preprocessing pipeline removing Twitter `@user` handles while preserving support URLs and technical tokens (e.g. `iOS 11.2`, `iPhone 8`).
- **Why Made**: Raw user handles create noisy vocabulary features that lead to spurious correlations during TF-IDF vectorization.
- **Alternative Considered**: Raw text tokenization without handle stripping.
- **Why Rejected**: Inspection of raw vector features showed user handles dominating the top 100 TF-IDF features, hurting classifier generalization.

---

### Decision 3: 10 + 1 Intent Taxonomy Design
- **Decision**: Define a 10 specific technical domain intent taxonomy (`SOFTWARE_UPDATE_ISSUES`, `BATTERY_POWER_ISSUES`, `DEVICE_FREEZE_REBOOT`, `ACCOUNT_APPLE_ID_SECURITY`, `ICLOUD_STORAGE_SYNC`, `BILLING_APP_STORE_SUBSCRIPTION`, `CONNECTIVITY_NETWORK_BLUETOOTH`, `HARDWARE_DISPLAY_AUDIO`, `HARDWARE_REPAIR_SERVICE_INQUIRY`, `GENERAL_HOW_TO_INFO`) plus 1 catch-all (`OTHER_AMBIGUOUS_COMPLAINT`).
- **Why Made**: Balances technical granularity with data support. Each category has explicit inclusion/exclusion criteria and default handling rules.
- **Alternative Considered**: Generic binary classification (`AUTO_HANDLE` vs `ESCALATE`) or fine-grained 50+ topic cluster labels.
- **Why Rejected**: Binary classification provides zero explainability for why a decision was made. A 50+ class taxonomy suffers from severe class overlap and high variance.

---

### Decision 4: Strict 90/10 Retrieval Split & Leakage Filter
- **Decision**: Partition reconstructed conversation pairs into a 94,098-example Training Retrieval Corpus and a 10,456-example Evaluation Pool. Enforce an explicit runtime filter excluding `customer_tweet_id` during similarity retrieval.
- **Why Made**: Prevents evaluation data leakage where test queries retrieve their exact historical match.
- **Alternative Considered**: Searching the entire dataset during evaluation without ID exclusion.
- **Why Rejected**: Searching test items against themselves artificially inflates retrieval similarity to 1.0, creating invalid evaluation results.

---

### Decision 5: Multi-Signal Safety & Escalation Policy
- **Decision**: Design a deterministic 4-stage escalation policy:
  1. Mandatory Sensitive Intent Check (Account, Billing, Hardware)
  2. Intent Classifier Confidence Check (< 0.35 threshold)
  3. Retrieval Evidence Quality Check (< 0.25 similarity score threshold)
  4. Safe Auto-Handle Execution
- **Why Made**: Guarantees safety and privacy compliance before executing auto-responses.
- **Alternative Considered**: End-to-end ML model trained directly on auto vs escalate labels.
- **Why Rejected**: End-to-end classifiers can produce high confidence on adversarial or subtle security prompts, creating dangerous security vulnerabilities.

---

### Decision 6: TF-IDF + Cosine Similarity Retrieval Engine
- **Decision**: Implement a fast, in-memory TF-IDF vectorizer and cosine similarity index over 94,098 historical pairs.
- **Why Made**: Delivers sub-millisecond retrieval latency (< 1ms), deterministic similarity scoring, and zero external vector database dependencies or API costs.
- **Alternative Considered**: Dense neural embeddings (e.g. OpenAI embeddings, ChromaDB / FAISS).
- **Why Rejected**: Dense vector indexes introduce high memory overhead, slower index build times, and non-deterministic scores, while TF-IDF excels at keyword matching for specific technical terms (e.g., `iOS 11.0.1`, `error 4013`, `AirPods`).

---

### Decision 7: Stratified 200-Example Golden Evaluation Set
- **Decision**: Construct a 200-example Golden Set with equal representation (~18-20 per intent class), difficulty tags, reference resolution points, and explicit human ground truth escalation labels.
- **Why Made**: Provides a high-quality, balanced benchmark that tests rare critical categories (Account Security, Billing) equally with high-frequency categories (Software Updates).
- **Alternative Considered**: Unstratified random sampling of 50-100 test items.
- **Why Rejected**: Random sampling resulted in 70%+ software update queries, underrepresenting security/billing queries and masking safety failures.

---

### Decision 8: Dual-Mode Reply Generator (LLM + Grounded Template Fallback)
- **Decision**: Implement GroundedReplyGenerator with support for Google Gemini API and a deterministic template fallback that extracts exact support links from historical evidence.
- **Why Made**: Ensures full execution and evaluation capability even when API keys are unconfigured or rate-limited.
- **Alternative Considered**: Strict dependency on external LLM calls.
- **Why Rejected**: Network outages or missing API keys would halt execution and evaluation pipelines.

---

### Decision 9: "Dangerous Auto-Handle Rate" as Core Primary Safety Metric
- **Decision**: Establish *Dangerous Auto-Handle Rate* (fraction of auto-handled queries that actually required human escalation) as the primary safety metric.
- **Why Made**: Auto-handling an account security or billing issue can lead to account compromise or financial loss, whereas false escalations merely add minor agent load.
- **Alternative Considered**: Raw overall accuracy or overall auto-handle rate.
- **Why Rejected**: Overall accuracy can hide dangerous security failures behind high performance on trivial queries.

---

### Decision 10: LLM-as-Judge Evaluation Framework
- **Decision**: Evaluate generated response quality across Groundedness, Intent Alignment, and Tone & Safety using structured LLM evaluation prompts supplemented by rule-based heuristic checks.
- **Why Made**: Standard n-gram metrics (BLEU, ROUGE) fail to measure factual accuracy and tone in customer support conversations.
- **Alternative Considered**: Relying solely on BLEU / ROUGE n-gram overlap scores.
- **Why Rejected**: BLEU penalizes correctly rephrased responses and rewards irrelevant responses that happen to share boilerplate words.

---

### Decision 11: Keyword-Heuristic Weak Supervision for Corpus Labeling
- **Decision**: Use domain keyword rules derived from empirical taxonomy analysis to label the 94,098 training corpus for fitting the TF-IDF intent classifier.
- **Why Made**: Enables training an intent classifier over a massive dataset without thousands of dollars in manual or LLM labeling costs.
- **Alternative Considered**: LLM zero-shot labeling of all 94,098 training items.
- **Why Rejected**: Costly ($150+ in API calls) and prohibitively slow (hours of API calls).

---

### Decision 12: Mandatory DM Escalation Pathway for Sensitive Queries
- **Decision**: Automatically route all account, security, and billing queries to direct private message (DM) links (`https://t.co/AppleSupportDM`).
- **Why Made**: Public Twitter support must NEVER ask for or process customer credentials or personal details in public tweets.
- **Alternative Considered**: Responding with public troubleshooting steps for account lockouts.
- **Why Rejected**: Directing users to troubleshoot account lockouts publicly creates security risks and violates Apple's social media support protocol.
