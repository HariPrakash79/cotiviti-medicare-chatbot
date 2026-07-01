"""
Medicare DME Policy Assistant v2.0
An advanced RAG chatbot with hybrid retrieval, cross-encoder re-ranking,
guardrails, and conversation context for answering Medicare coverage
policy questions about glucose monitors (BGM & CGM).

Built as a Proof of Concept for Cotiviti Intern Assessment.
Topic 3: Content Management in Health Care

Architecture:
    User Query
        -> [Guardrails: Input Validation & Topic Scoping]
        -> [Query Classification: coverage / billing / documentation / general]
        -> [Hybrid Retrieval: BM25 (keyword) + FAISS (semantic)]
        -> [Reciprocal Rank Fusion: merge results]
        -> [Cross-Encoder Re-ranking: precision pass]
        -> [Conversation Context: include chat history for follow-ups]
        -> [LLM Generation: type-specific prompt to Gemini]
        -> [Guardrails: Confidence Scoring & Output Validation]
        -> Streamlit Chat UI with sources + confidence
"""

import streamlit as st
import os
import re
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
from google import genai
from google.genai import types as genai_types

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
PDF_DIR = Path(__file__).parent / "files"

PDF_FILES = {
    "LCD - Glucose Monitors (L33822).pdf":
        "LCD L33822 – Glucose Monitors",
    "Article - Glucose Monitor - Policy Article (A52464).pdf":
        "Policy Article A52464 – Glucose Monitor",
    "Article - Standard Documentation Requirements for All Claims Submitted to DME MACs (A55426).pdf":
        "Standard Documentation Article A55426",
}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SEMANTIC_TOP_K = 15
BM25_TOP_K = 15
RERANK_TOP_K = 6
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-2.5-flash"


# ═══════════════════════════════════════════════════════════════════════════
# 1. GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════
class Guardrails:
    """Input validation, topic scoping, and output quality assessment."""

    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
        r"disregard\s+(your|all|the)\s+(rules|instructions|guidelines)",
        r"you\s+are\s+now\s+a",
        r"pretend\s+(you\s+are|to\s+be)",
        r"forget\s+(everything|all|your)",
        r"new\s+system\s+prompt",
        r"override\s+(system|safety|your)",
        r"jailbreak",
        r"\bDAN\b",
    ]

    TOPIC_KEYWORDS = [
        "medicare", "medicaid", "cms", "coverage", "covered", "glucose", "monitor",
        "bgm", "cgm", "diabetes", "diabetic", "insulin", "hcpcs", "icd",
        "dme", "dmepos", "supplier", "beneficiary", "claim", "billing",
        "code", "modifier", "lancet", "test strip", "prescription",
        "swo", "wopd", "pod", "medical necessity", "reasonable",
        "necessary", "lcd", "policy", "article", "documentation",
        "refill", "delivery", "equipment", "supply", "allowance",
        "continuous", "adjunctive", "non-adjunctive", "blood",
        "a4238", "a4239", "a4253", "a4259", "a4258", "a4257",
        "a4256", "a4271", "a4250", "a4244", "a4245", "a9270", "a9275",
        "e2102", "e2103", "e0607", "e2100", "e2101", "e2104", "e0620",
        "face-to-face", "telehealth", "practitioner", "physician",
        "hypoglycemia", "glycemic", "treatment", "sensor", "transmitter",
        "dexterity", "impairment", "visual", "acuity", "spring",
        "dispensing", "utilization", "denied", "non-covered",
        "maintenance", "routine", "laser", "piercing", "device",
        "powered", "disposable",
    ]

    @classmethod
    def validate_input(cls, query: str) -> tuple[bool, str]:
        if not query or len(query.strip()) < cls.MIN_QUERY_LENGTH:
            return False, "Query is too short. Please ask a complete question."
        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, f"Query exceeds {cls.MAX_QUERY_LENGTH} characters. Please shorten it."
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains disallowed patterns. Please rephrase."
        return True, ""

    @classmethod
    def check_topic_relevance(cls, query: str) -> tuple[bool, float]:
        query_lower = query.lower()
        matches = sum(1 for kw in cls.TOPIC_KEYWORDS if kw in query_lower)
        score = min(matches / 3.0, 1.0)
        is_relevant = score >= 0.33 or len(query.split()) <= 5
        return is_relevant, score

    @classmethod
    def assess_confidence(cls, retrieval_scores: list[float]) -> dict:
        if not retrieval_scores:
            return {"level": "low", "score": 0, "pct": "0%", "color": "red"}
        avg = sum(retrieval_scores) / len(retrieval_scores)
        top = max(retrieval_scores)
        combined = 0.6 * top + 0.4 * avg
        # Normalize cross-encoder scores (typically -12 to +12) to 0-100%
        pct = max(0, min(100, int((combined + 5) / 15 * 100)))
        if pct >= 60:
            return {"level": "high", "score": pct, "pct": f"{pct}%", "color": "green"}
        if pct >= 35:
            return {"level": "medium", "score": pct, "pct": f"{pct}%", "color": "orange"}
        return {"level": "low", "score": pct, "pct": f"{pct}%", "color": "red"}


# ═══════════════════════════════════════════════════════════════════════════
# 2. SMART CHUNKING (Section-Aware)
# ═══════════════════════════════════════════════════════════════════════════
SECTION_HEADER_PATTERNS = [
    r"^(HOME BLOOD GLUCOSE MONITORS.*)",
    r"^(CONTINUOUS GLUCOSE MONITORS.*)",
    r"^(GENERAL\b.*)",
    r"^(REFILL REQUIREMENTS.*)",
    r"^(DOCUMENTATION REQUIREMENTS.*)",
    r"^(POLICY SPECIFIC DOCUMENTATION.*)",
    r"^(STANDARD WRITTEN ORDER.*)",
    r"^(WRITTEN ORDERS? PRIOR TO DELIVERY.*)",
    r"^(PROOF OF DELIVERY.*)",
    r"^(FACE-TO-FACE ENCOUNTER.*)",
    r"^(REPAIRS?/REPLACEMENT.*)",
    r"^(REPLACEMENT\b.*)",
    r"^(REPAIRS?\b.*)",
    r"^(CODING GUIDELINES.*)",
    r"^(NON-MEDICAL NECESSITY COVERAGE.*)",
    r"^(Coverage Guidance.*)",
    r"^(Coverage Indications.*)",
    r"^(Usual Utilization.*)",
    r"^(High Utilization.*)",
    r"^(CGM Continued Coverage.*)",
    r"^(CLAIM NARRATIVES.*)",
    r"^(DATE SPANS ON CLAIMS.*)",
    r"^(CONTINUED MEDICAL NEED.*)",
    r"^(CONTINUED USE.*)",
    r"^(NEW ORDER REQUIREMENTS.*)",
    r"^(EQUIPMENT RETAINED.*)",
    r"^(REASONABLE AND NECESSARY.*)",
    r"^(MEDICAL RECORD DOCUMENTATION.*)",
    r"^(SIGNATURE REQUIREMENTS.*)",
]


def detect_section(line: str) -> str | None:
    stripped = line.strip()
    for pattern in SECTION_HEADER_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return stripped
    if stripped.isupper() and 5 < len(stripped) < 120 and not stripped.startswith("E0") and not stripped.startswith("A4"):
        return stripped
    return None


def extract_pdf_text(pdf_path: Path) -> str:
    with pdfplumber.open(str(pdf_path)) as pdf:
        return "\n\n".join(p.extract_text() or "" for p in pdf.pages)


def smart_chunk(text: str, source: str) -> list[dict]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = text.split("\n")
    chunks = []
    current_section = "General"
    current_text = ""

    for line in lines:
        header = detect_section(line)
        if header:
            if current_text.strip():
                for chunk in _split_if_large(current_text.strip(), source, current_section):
                    chunks.append(chunk)
            current_section = header
            current_text = line + "\n"
        else:
            current_text += line + "\n"

    if current_text.strip():
        for chunk in _split_if_large(current_text.strip(), source, current_section):
            chunks.append(chunk)

    return chunks


def _split_if_large(text: str, source: str, section: str) -> list[dict]:
    if len(text) <= CHUNK_SIZE:
        return [{"text": text, "source": source, "section": section}]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append({"text": current, "source": source, "section": section})
            if len(para) > CHUNK_SIZE:
                words = para.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 > CHUNK_SIZE:
                        chunks.append({"text": current.strip(), "source": source, "section": section})
                        current = ""
                    current = f"{current} {word}" if current else word
            else:
                overlap_src = current[-(CHUNK_OVERLAP):] if current and len(current) > CHUNK_OVERLAP else ""
                current = f"{overlap_src}\n\n{para}" if overlap_src else para

    if current.strip():
        chunks.append({"text": current.strip(), "source": source, "section": section})

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUERY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
QUERY_TYPES = {
    "coverage": {
        "keywords": [
            "cover", "eligible", "criteria", "qualify", "approved",
            "medical necessity", "reasonable and necessary", "denied",
            "indications", "limitations", "insulin", "non-insulin",
            "adjunctive", "non-adjunctive", "cgm", "bgm",
        ],
        "label": "Coverage Criteria",
        "color": "#2196F3",
    },
    "billing": {
        "keywords": [
            "bill", "code", "hcpcs", "modifier", "claim", "a4238",
            "a4239", "e2102", "e2103", "strip", "lancet", "supply",
            "allowance", "quantity", "unit of service", "uos",
            "cg", "kf", "ks", "kx", "ey",
        ],
        "label": "Billing & Coding",
        "color": "#FF9800",
    },
    "documentation": {
        "keywords": [
            "document", "swo", "written order", "wopd", "pod",
            "proof of delivery", "prescription", "signature",
            "face-to-face", "medical record", "refill", "order",
            "attestation", "narrative",
        ],
        "label": "Documentation",
        "color": "#4CAF50",
    },
}


def classify_query(query: str) -> tuple[str, str, str]:
    query_lower = query.lower()
    scores = {}
    for qtype, info in QUERY_TYPES.items():
        score = sum(1 for kw in info["keywords"] if kw in query_lower)
        scores[qtype] = score

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general", "General", "#9E9E9E"
    return best, QUERY_TYPES[best]["label"], QUERY_TYPES[best]["color"]


# ═══════════════════════════════════════════════════════════════════════════
# 4. HYBRID RETRIEVER (BM25 + FAISS + Cross-Encoder Re-ranking)
# ═══════════════════════════════════════════════════════════════════════════
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_NAME)


@st.cache_resource(show_spinner=False)
def build_knowledge_base():
    embed_model = load_embed_model()
    all_chunks = []

    for filename, display_name in PDF_FILES.items():
        path = PDF_DIR / filename
        if not path.exists():
            continue
        text = extract_pdf_text(path)
        chunks = smart_chunk(text, display_name)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None, None, []

    # FAISS index (semantic)
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    # BM25 index (keyword)
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    return faiss_index, bm25_index, all_chunks


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rlist in ranked_lists:
        for rank, doc_id in enumerate(rlist):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query: str,
    faiss_index,
    bm25_index,
    chunks: list[dict],
    embed_model,
    cross_encoder,
    semantic_k: int = SEMANTIC_TOP_K,
    bm25_k: int = BM25_TOP_K,
    final_k: int = RERANK_TOP_K,
) -> list[dict]:

    # Stage 1a: FAISS semantic search
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    _, faiss_ids = faiss_index.search(q_emb, semantic_k)
    faiss_ranked = [int(i) for i in faiss_ids[0] if 0 <= i < len(chunks)]

    # Stage 1b: BM25 keyword search
    tokenized_query = tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked = list(np.argsort(bm25_scores)[::-1][:bm25_k])

    # Stage 2: Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([faiss_ranked, bm25_ranked])
    candidate_ids = [doc_id for doc_id, _ in fused[: semantic_k + bm25_k]]

    # Stage 3: Cross-encoder re-ranking
    if not candidate_ids:
        return []

    unique_ids = list(dict.fromkeys(candidate_ids))
    pairs = [[query, chunks[i]["text"]] for i in unique_ids]
    ce_scores = cross_encoder.predict(pairs)

    scored = list(zip(unique_ids, ce_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in scored[:final_k]:
        results.append({
            **chunks[doc_id],
            "score": float(score),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONVERSATION CONTEXT
# ═══════════════════════════════════════════════════════════════════════════
def build_conversation_context(messages: list[dict], max_turns: int = 3) -> str:
    recent = []
    turns = 0
    for msg in reversed(messages):
        if turns >= max_turns:
            break
        recent.insert(0, msg)
        if msg["role"] == "user":
            turns += 1

    if not recent:
        return ""

    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def expand_follow_up(query: str, history: list[dict]) -> str:
    follow_up_signals = [
        "what about", "how about", "and for", "what if",
        "can you also", "tell me more", "elaborate",
        "same question", "similarly",
    ]
    is_short = len(query.split()) <= 6
    has_signal = any(s in query.lower() for s in follow_up_signals)
    has_pronoun = bool(re.search(r"\b(it|its|they|them|their|this|that|these|those)\b", query, re.I))

    if (is_short or has_signal or has_pronoun) and history:
        for msg in reversed(history):
            if msg["role"] == "user":
                return f"{msg['content']} — follow-up: {query}"
        return query
    return query


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM GENERATION (Type-Specific Prompts)
# ═══════════════════════════════════════════════════════════════════════════
BASE_SYSTEM_PROMPT = """You are a Medicare coverage policy expert assistant specializing in Durable Medical Equipment (DME), specifically glucose monitors (BGM and CGM).

You help healthcare providers, suppliers, and beneficiaries understand Medicare coverage rules, billing requirements, and documentation standards.

STRICT RULES:
1. Answer ONLY from the provided policy excerpts. NEVER use outside knowledge.
2. If the excerpts do not contain the answer, state: "This information is not available in the loaded policy documents."
3. Cite the source document for every factual claim.
4. Be precise with HCPCS codes, ICD-10 codes, modifiers, quantities, and timeframes.
5. When listing criteria, include ALL of them — do not omit any.
6. Do NOT provide medical advice — only explain what the policy documents state."""

TYPE_SPECIFIC_INSTRUCTIONS = {
    "coverage": """
ADDITIONAL INSTRUCTIONS FOR COVERAGE QUESTIONS:
- List ALL coverage criteria numbered exactly as they appear in the policy.
- Clearly distinguish between initial coverage and continued coverage criteria.
- Note any differences between BGM and CGM coverage.
- Specify which conditions lead to denial.""",

    "billing": """
ADDITIONAL INSTRUCTIONS FOR BILLING/CODING QUESTIONS:
- List exact HCPCS codes with their full descriptions.
- Specify required modifiers (CG, KF, KS, KX, EY) and when each applies.
- Include quantities, units of service, and time periods.
- Note any bundling rules or unbundling restrictions.""",

    "documentation": """
ADDITIONAL INSTRUCTIONS FOR DOCUMENTATION QUESTIONS:
- List all required elements of the document being asked about.
- Specify timeframes (e.g., "within 6 months", "7 years retention").
- Distinguish between required vs. recommended documentation.
- Note any differences between initial and refill/renewal documentation.""",

    "general": """
ADDITIONAL INSTRUCTIONS:
- Provide a comprehensive answer covering all relevant aspects.
- If the question spans multiple topics, organize by subtopic.""",
}


def generate_response(
    query: str,
    context_chunks: list[dict],
    api_key: str,
    query_type: str,
    conversation_ctx: str,
) -> str:
    client = genai.Client(api_key=api_key)

    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Section: {c.get('section', 'N/A')}]\n{c['text']}"
        for c in context_chunks
    )

    type_instructions = TYPE_SPECIFIC_INSTRUCTIONS.get(query_type, "")

    conv_block = ""
    if conversation_ctx:
        conv_block = f"\nRECENT CONVERSATION (for context on follow-up questions):\n{conversation_ctx}\n"

    prompt = f"""POLICY DOCUMENT EXCERPTS:
{context_block}
{conv_block}
USER QUESTION:
{query}

ANSWER:"""

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=f"{BASE_SYSTEM_PROMPT}{type_instructions}",
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    # resp.text raises ValueError if the response was blocked by safety filters
    if not resp.candidates:
        raise RuntimeError("Gemini returned no candidates (response may have been blocked)")
    candidate = resp.candidates[0]
    if not candidate.content or not candidate.content.parts:
        finish = getattr(candidate, "finish_reason", "unknown")
        raise RuntimeError(f"Gemini response empty (finish_reason={finish})")
    return resp.text


# ═══════════════════════════════════════════════════════════════════════════
# 7. STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════
def render_ui():
    st.set_page_config(
        page_title="Medicare DME Policy Assistant",
        page_icon=":hospital:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { max-width: 1000px; }
        .source-card {
            background: #f8f9fb;
            border-left: 3px solid #4a7dfc;
            padding: 0.6rem 0.8rem;
            margin: 0.3rem 0;
            border-radius: 4px;
            font-size: 0.84rem;
            line-height: 1.45;
        }
        .source-card strong { color: #2c3e50; }
        .confidence-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            color: white;
        }
        .query-type-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            color: white;
            margin-right: 6px;
        }
        .pipeline-step {
            padding: 0.3rem 0.6rem;
            margin: 0.15rem 0;
            border-radius: 4px;
            font-size: 0.82rem;
        }
        div[data-testid="stSidebar"] .stMarkdown h1 { font-size: 1.3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Medicare DME Policy Assistant")
        st.caption("Advanced RAG Chatbot — Cotiviti POC v2.0")

        env_api_key = os.getenv("GEMINI_API_KEY", "")
        if env_api_key:
            api_key = env_api_key
        else:
            st.subheader("API Key")
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Free key: https://aistudio.google.com/apikey",
            )
            if not api_key:
                st.info("Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)")
            else:
                st.success("Key loaded")

        st.divider()
        st.subheader("Knowledge Base")
        found = 0
        for filename, display_name in PDF_FILES.items():
            exists = (PDF_DIR / filename).exists()
            icon = ":white_check_mark:" if exists else ":x:"
            st.markdown(f"{icon} {display_name}")
            found += int(exists)

        st.divider()
        st.subheader("RAG Pipeline")
        st.markdown(
            "1. **Guardrails** — input validation & topic scoping\n"
            "2. **Query Classification** — coverage / billing / docs\n"
            "3. **Hybrid Retrieval** — BM25 + FAISS semantic search\n"
            "4. **Rank Fusion** — Reciprocal Rank Fusion (RRF)\n"
            "5. **Re-ranking** — Cross-encoder precision pass\n"
            "6. **Generation** — Gemini with type-specific prompt\n"
            "7. **Confidence Scoring** — retrieval quality check"
        )

        st.divider()
        st.subheader("Models")
        st.markdown(
            f"- **Embeddings:** `{EMBED_MODEL_NAME}`\n"
            f"- **Re-ranker:** `{CROSS_ENCODER_NAME}`\n"
            f"- **LLM:** `{GEMINI_MODEL}`"
        )

        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # ── Main Area ──────────────────────────────────────────────────────────
    st.header("Medicare Glucose Monitor Coverage Q&A")
    st.caption(
        "Ask questions about BGM/CGM coverage criteria, billing codes, "
        "documentation requirements, and supplier guidelines."
    )

    # Load models and build KB
    with st.spinner("Loading models and building knowledge base..."):
        faiss_index, bm25_index, chunks = build_knowledge_base()
        embed_model = load_embed_model()
        cross_enc = load_cross_encoder()

    if faiss_index is None or not chunks:
        st.error("No policy PDFs found. Place them in the same folder as this script.")
        return

    st.success(f"Ready — {len(chunks)} passages indexed from {found} documents  |  Hybrid retrieval + cross-encoder re-ranking active")

    # ── Starter Questions ──────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("#### Try one of these questions:")
        suggestions = [
            "What are the 5 initial coverage criteria for a CGM?",
            "How many test strips are covered for insulin vs non-insulin patients?",
            "What modifiers (CG, KF, KS, KX) are required on glucose monitor claims?",
            "What elements must a Standard Written Order (SWO) contain?",
            "What is the difference between adjunctive and non-adjunctive CGMs?",
            "What are the refill documentation requirements for DMEPOS supplies?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(q, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

    # ── Chat History ───────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta")
            if meta:
                conf = meta.get("confidence", {})
                qtype_label = meta.get("query_type_label", "")
                qtype_color = meta.get("query_type_color", "#9E9E9E")
                conf_color_hex = {"green": "#4CAF50", "orange": "#FF9800", "red": "#f44336"}.get(conf.get("color", "red"), "#9E9E9E")

                st.markdown(
                    f'<span class="query-type-badge" style="background:{qtype_color}">{qtype_label}</span>'
                    f'<span class="confidence-badge" style="background:{conf_color_hex}">Confidence: {conf.get("level", "?")} ({conf.get("pct", "?")})</span>',
                    unsafe_allow_html=True,
                )
            if msg.get("sources"):
                with st.expander("View source references"):
                    for src in msg["sources"]:
                        section = src.get("section", "")
                        section_tag = f" | Section: {section}" if section else ""
                        st.markdown(
                            f'<div class="source-card">'
                            f"<strong>{src['source']}{section_tag}</strong> "
                            f"(re-rank score: {src['score']:.3f})<br>"
                            f"{src['text'][:400]}{'...' if len(src['text']) > 400 else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # ── Handle Input ───────────────────────────────────────────────────────
    query = st.chat_input("Ask about Medicare glucose monitor coverage...")

    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")

    if not query:
        return

    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar to get started.")
        return

    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Pipeline Execution ─────────────────────────────────────────────────
    with st.chat_message("assistant"):
        pipeline_container = st.container()

        # Step 1: Guardrails - Input
        valid, msg = Guardrails.validate_input(query)
        if not valid:
            st.warning(f"Input rejected: {msg}")
            st.session_state.messages.append({"role": "assistant", "content": f"Input rejected: {msg}"})
            return

        relevant, relevance_score = Guardrails.check_topic_relevance(query)
        if not relevant:
            off_topic_msg = (
                "This question doesn't appear to be related to Medicare DME coverage "
                "for glucose monitors. I can only answer questions about Medicare "
                "coverage policies, billing codes, and documentation requirements "
                "for BGM and CGM devices. Please rephrase your question."
            )
            st.warning(off_topic_msg)
            st.session_state.messages.append({"role": "assistant", "content": off_topic_msg})
            return

        # Step 2: Query Classification
        qtype, qtype_label, qtype_color = classify_query(query)

        # Step 3: Expand follow-up queries
        retrieval_query = expand_follow_up(query, st.session_state.messages[:-1])

        # Step 4: Hybrid Retrieval + Re-ranking
        with st.spinner("Searching policy documents (BM25 + semantic + re-ranking)..."):
            t0 = time.time()
            context = hybrid_retrieve(
                retrieval_query, faiss_index, bm25_index, chunks,
                embed_model, cross_enc,
            )
            retrieval_time = time.time() - t0

        # Step 5: Confidence Assessment
        scores = [c["score"] for c in context]
        confidence = Guardrails.assess_confidence(scores)

        # Step 6: Build conversation context
        conv_ctx = build_conversation_context(st.session_state.messages[:-1])

        # Step 7: Generate
        with st.spinner("Generating answer..."):
            t0 = time.time()
            try:
                answer = generate_response(query, context, api_key, qtype, conv_ctx)
            except Exception as exc:
                answer = f"Error generating response: {exc}"
            gen_time = time.time() - t0

        # ── Display ───────────────────────────────────────────────────────
        st.markdown(answer)

        conf_color_hex = {"green": "#4CAF50", "orange": "#FF9800", "red": "#f44336"}.get(confidence["color"], "#9E9E9E")
        st.markdown(
            f'<span class="query-type-badge" style="background:{qtype_color}">{qtype_label}</span>'
            f'<span class="confidence-badge" style="background:{conf_color_hex}">'
            f'Confidence: {confidence["level"]} ({confidence["pct"]})</span>'
            f'&nbsp;&nbsp;<span style="font-size:0.78rem;color:#888;">Retrieval: {retrieval_time:.2f}s | Generation: {gen_time:.2f}s</span>',
            unsafe_allow_html=True,
        )

        if confidence["level"] == "low":
            st.warning(
                "Low confidence — the retrieved passages may not fully cover this question. "
                "Consider rephrasing or asking a more specific question."
            )

        with st.expander("View source references"):
            for src in context:
                section = src.get("section", "")
                section_tag = f" | Section: {section}" if section else ""
                st.markdown(
                    f'<div class="source-card">'
                    f"<strong>{src['source']}{section_tag}</strong> "
                    f"(re-rank score: {src['score']:.3f})<br>"
                    f"{src['text'][:400]}{'...' if len(src['text']) > 400 else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": context,
            "meta": {
                "confidence": confidence,
                "query_type": qtype,
                "query_type_label": qtype_label,
                "query_type_color": qtype_color,
                "retrieval_time": retrieval_time,
                "generation_time": gen_time,
            },
        })


if __name__ == "__main__":
    render_ui()
