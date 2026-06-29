# Medicare DME Policy Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about Medicare coverage policies for glucose monitors (BGM & CGM).

**Cotiviti Intern Assessment — Topic 3: Content Management in Health Care**

## Architecture

```
User Question
     |
     v
[Sentence-Transformers]  -->  Embed query
     |
     v
[FAISS Vector Index]     -->  Retrieve top-k relevant policy passages
     |
     v
[Google Gemini LLM]      -->  Generate accurate answer with citations
     |
     v
Streamlit Chat UI        -->  Display answer + source references
```

## Knowledge Base

The chatbot is grounded in three official CMS/Medicare policy documents:

1. **LCD L33822** — Glucose Monitors Local Coverage Determination
2. **Policy Article A52464** — Glucose Monitor Policy Article (coding guidelines, modifiers, ICD-10 codes)
3. **Standard Documentation Article A55426** — Documentation requirements for all DME MAC claims

## Setup

### Prerequisites
- Python 3.10+
- A free Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Installation

```bash
cd Chatbot
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens in your browser. Paste your Gemini API key in the sidebar and start asking questions.

## Example Questions

- What are the initial coverage criteria for a CGM?
- How many test strips and lancets are covered for insulin-treated patients?
- What modifiers are required on CGM claims?
- What must a Standard Written Order (SWO) contain?
- What is the difference between adjunctive and non-adjunctive CGMs?
- What are the high utilization criteria for BGM testing supplies?

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| LLM | Google Gemini 2.0 Flash |
| Language | Python 3.10+ |
