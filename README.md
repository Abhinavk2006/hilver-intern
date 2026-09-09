# Production-Minded AI Customer-Support Agent for @AppleSupport

An end-to-end, grounded AI customer-support agent built for **@AppleSupport** using the Twitter Customer Support dataset. The system performs empirical intent classification, historical RAG evidence retrieval, explicit multi-signal safety escalation, and grounded reply generation.

---

## Key Performance Highlights

- **Intent Classification Accuracy**: **90.0%** (Macro F1: **90.4%**) across an 11-category intent taxonomy.
- **Escalation Recall**: **96.7%** (catches 96.7% of queries requiring human/privacy intervention).
- **Dangerous Auto-Handle Rate**: **2.97%** (vs **45.5%** in naive 100% auto-handle baselines).
- **Sub-millisecond Latency**: Local TF-IDF classifier and vector retriever execute in < 10ms.

---

## System Architecture

```
[ Incoming Tweet ]
       │
       ▼
[ Stage 1: Preprocessing & Handle Normalization ]
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
[ Stage 2: TF-IDF Intent Classifier ]  [ Stage 3: Historical RAG Retriever ]
       │ (10+1 Categories)                      │ (94,098 Corpus Pairs)
       └───────────────────┬────────────────────┘
                           ▼
            [ Stage 4: Multi-Signal Escalation Policy ]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      [ ESCALATE ]                 [ AUTO_HANDLE ]
             │                           │
             ▼                           ▼
[ Polite DM / Genius Bar Link ]   [ Grounded Evidence Reply ]
```

---

## 15-Minute Reproducibility Guide

### Prerequisites
- Python 3.10+
- Windows / Linux / macOS environment

### 1. Environment Setup
```bash
# Clone repository and enter directory
cd "hilver intern"

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install pandas scikit-learn datasets google-genai numpy
```

### 2. Dataset Reconstruction & Golden Set Generation
*(Note: Processed data and Golden Set are already included in `data/processed/` and `data/golden_set/`)*
```bash
# To re-run dataset processing from scratch:
python scripts/fetch_data.py
python scripts/create_golden_set.py
```

### 3. Model Training & Indexing (< 1 minute)
Fits the TF-IDF Intent Classifier and builds the historical retrieval index over 94,098 conversation pairs:
```bash
python scripts/train_models.py
```

### 4. Run System Evaluation (< 1 minute)
Evaluates Baseline 1, Baseline 2, and the Main System across the 200-example Golden Set:
```bash
python scripts/evaluate.py
```

---

## Comparative Benchmark Results

| Metric | Baseline 1 (Trivial Auto-Handle) | Baseline 2 (TF-IDF + Policy) | Main System (Full Pipeline) |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 9.50% | **90.00%** | **90.00%** |
| **Macro F1 Score** | 0.0158 | **0.9041** | **0.9041** |
| **Auto-Handle Rate** | 100.00% | 50.50% | 50.50% |
| **Escalation Recall** | 0.00% | **96.70%** | **96.70%** |
| **Dangerous Auto-Handle Count** | 91 / 200 (45.5%) | **3 / 101 (2.97%)** | **3 / 101 (2.97%)** |
| **Avg Reply Quality (0-10)** | 10.00 | **9.97** | **9.97** |

---

## Directory Structure

```
hilver intern/
├── data/
│   ├── golden_set/
│   │   └── golden_set.json           # 200-example hand-labeled golden evaluation set
│   └── processed/
│       ├── apple_support_pairs.json  # 104,554 reconstructed AppleSupport conversations
│       ├── train_retrieval_corpus.json # 94,098 retrieval training pairs
│       └── test_evaluation_set.json  # 10,456 test pool pairs
├── results/
│   ├── tfidf_intent_model.pkl       # Fitted TF-IDF + Logistic Regression model
│   ├── retrieval_index.pkl          # Indexed historical vector corpus
│   └── evaluation_summary.json      # Full empirical evaluation output & failure logs
├── scripts/
│   ├── fetch_data.py                 # Reconstructs AppleSupport conversation pairs
│   ├── create_golden_set.py          # Samples and stratifies 200 golden set entries
│   ├── train_models.py               # Fits intent model and builds retrieval index
│   └── evaluate.py                   # Runs 3-baseline comparative evaluation
├── src/
│   ├── agent.py                      # End-to-end agent orchestrator
│   ├── classifier.py                 # Intent classification module
│   ├── config.py                     # Global paths & hyperparameter configs
│   ├── conversation.py               # Reconstructs parent-child tweet threads
│   ├── data_loader.py                # HuggingFace datasets loader
│   ├── escalation.py                 # Multi-signal safety escalation policy
│   ├── evaluation.py                 # Comprehensive evaluation harness
│   ├── generator.py                  # Grounded reply generator (LLM + Template fallback)
│   ├── intents.py                    # 10+1 Intent taxonomy definition
│   ├── preprocessing.py              # Text cleaning & handle normalization
│   └── retriever.py                  # Leakage-free TF-IDF cosine retriever
├── decision_log.md                   # 12 explicit architectural decisions & trade-offs
├── report.md                         # Detailed technical report & empirical analysis
└── README.md                         # This file
```

---

## Optional: Gemini LLM Integration
To enable LLM generation and LLM-as-judge scoring, export your API key:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# Linux/macOS
export GEMINI_API_KEY="your_api_key_here"
```
*(If no API key is provided, the system seamlessly uses deterministic grounded template generation).*
