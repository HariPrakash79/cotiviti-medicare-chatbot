# Medicare DME Policy Assistant

A production-grade RAG (Retrieval-Augmented Generation) chatbot that answers questions about Medicare coverage policies for Durable Medical Equipment (DME) — specifically blood glucose monitors (BGM) and continuous glucose monitors (CGM).

**Cotiviti Intern Assessment | Topic 3: Content Management in Health Care**

---

## Architecture

```
User Query
  |
  v
[Guardrails] -----------> Input validation, prompt injection detection, topic scoping
  |
  v
[Query Classifier] -----> Coverage / Billing & Coding / Documentation / General
  |
  v
[Hybrid Retrieval]
  |-- BM25 (keyword) ---> Exact matches for HCPCS codes, modifiers, quantities
  |-- FAISS (semantic) -> Meaning-based passage matching (all-MiniLM-L6-v2)
  |
  v
[Reciprocal Rank Fusion]  Merge and deduplicate results from both retrievers
  |
  v
[Cross-Encoder Rerank] -> ms-marco-MiniLM-L-6-v2 precision pass (top 8)
  |
  v
[Conversation Context] -> Follow-up detection + chat history injection
  |
  v
[Groq LLM] -------------> Llama 3.3 70B with type-specific system prompts
  |
  v
[Confidence Scoring] ----> Retrieval quality assessment (shown as %)
  |
  v
[FastAPI + Custom UI] ---> HTML/Tailwind/JS frontend with source citations
```

## Knowledge Base

Grounded in three official CMS/Medicare policy documents:

| Document | ID | Content |
|---|---|---|
| Local Coverage Determination | LCD L33822 | BGM/CGM coverage criteria, utilization limits, refill rules |
| Policy Article | A52464 | Coding guidelines, HCPCS modifiers (CG/KF/KS/KX), 461 ICD-10 codes |
| Standard Documentation Article | A55426 | SWO, WOPD, POD requirements, face-to-face encounters, repairs/replacement |

## Key Features

- **Hybrid Search** — BM25 keyword + FAISS semantic retrieval with Reciprocal Rank Fusion
- **Cross-Encoder Re-ranking** — Precision pass using ms-marco-MiniLM for accurate passage selection
- **Guardrails** — Input validation, prompt injection detection, topic scoping, confidence scoring
- **Query Classification** — Automatically detects question type and applies specialized prompts
- **Conversation Memory** — Handles follow-up questions using chat history context
- **Source Citations** — Every answer shows the source document, section, and re-rank score
- **Confidence Scoring** — Displays retrieval quality as a percentage with color-coded badges

## Setup

### Prerequisites
- Python 3.10+
- A free Groq API key ([get one here](https://console.groq.com/keys))

### Installation

```bash
cd Chatbot
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `Chatbot/` directory:

```
GROQ_API_KEY=gsk_your_key_here
```

Or enter the key directly in the sidebar when running the app.

### Run

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Testing

A 55-question test suite validates answer accuracy and citation correctness across all policy categories, including 6 out-of-context guardrail tests.

```bash
python run_tests.py              # run all 55 questions
python run_tests.py --ids 1 5 9  # run specific questions
```

### Test Results

| Metric | Score |
|---|---|
| Overall Pass Rate | 37/50 (74%) |
| Keyword Accuracy | 73.2% avg |
| Citation Accuracy | 94.0% |
| Guardrail Test | PASS |

**Category Breakdown:**

| Category | Pass Rate | Keyword % | Citation % |
|---|---|---|---|
| BGM Utilization | 4/4 (100%) | 100% | 100% |
| Modifiers | 3/3 (100%) | 100% | 100% |
| Refill Requirements | 3/3 (100%) | 100% | 100% |
| Non-Medical Necessity | 2/2 (100%) | 100% | 100% |
| Coding | 4/6 (67%) | 75% | 100% |
| Documentation - SWO | 2/2 (100%) | 80% | 100% |
| Face-to-Face | 2/2 (100%) | 83% | 100% |
| Repairs/Replacement | 2/2 (100%) | 83% | 100% |

## Example Questions

- What are the 5 initial coverage criteria for a CGM?
- How many test strips are covered for insulin vs non-insulin patients?
- What modifiers (CG, KF, KS, KX) are required on glucose monitor claims?
- What elements must a Standard Written Order (SWO) contain?
- What is the difference between adjunctive and non-adjunctive CGMs?
- What are the refill documentation requirements for DMEPOS supplies?

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | HTML5 + Tailwind CSS + vanilla JavaScript |
| PDF Parsing | PyPDF2 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Keyword Search | BM25 (rank-bm25) |
| Vector Search | FAISS |
| Re-ranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| LLM | Groq — Llama 3.3 70B Versatile |
| Language | Python 3.10+ |

## Project Structure

```
Chatbot/
|-- main.py                # FastAPI backend + RAG pipeline
|-- app.py                 # Streamlit version (alternative)
|-- static/
|   |-- index.html         # Custom frontend
|-- files/
|   |-- LCD - Glucose Monitors (L33822).pdf
|   |-- Article - Glucose Monitor - Policy Article (A52464).pdf
|   |-- Article - Standard Documentation Requirements... (A55426).pdf
|-- run_tests.py           # Automated test runner
|-- test_questions.json    # 50 test Q&A pairs
|-- test_results.json      # Latest test results
|-- requirements.txt
|-- .env                   # API key (gitignored)
|-- .gitignore
|-- README.md
```
