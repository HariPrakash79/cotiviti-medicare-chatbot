"""
Medicare DME Policy Assistant v2.0
FastAPI backend with hybrid RAG pipeline, guardrails, and Gemini LLM.

Architecture:
    Query -> Guardrails -> Classify -> Hybrid Retrieve (BM25+FAISS)
          -> RRF Fusion -> Cross-Encoder Rerank -> Conversation Context
          -> Gemini 2.5 Flash -> Confidence Score -> Response
"""

from __future__ import annotations

import os
import re
import time
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
from google import genai
from google.genai import types as genai_types

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "files"
STATIC_DIR = BASE_DIR / "static"

PDF_FILES = {
    "LCD - Glucose Monitors (L33822).pdf":
        "LCD L33822 – Glucose Monitors",
    "Article - Glucose Monitor - Policy Article (A52464).pdf":
        "Policy Article A52464 – Glucose Monitor",
    "Article - Standard Documentation Requirements for All Claims Submitted to DME MACs (A55426).pdf":
        "Standard Documentation Article A55426",
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
RERANK_TOP_K = 8

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

load_dotenv(BASE_DIR / ".env")

# Global state populated at startup
state: dict = {}


# ═══════════════════════════════════════════════════════════════════════════
# 1. GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════
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
    "hcpcs code", "icd code", "modifier", "lancet", "test strip",
    "prescription", "swo", "wopd", "pod", "medical necessity",
    "reasonable", "necessary", "lcd", "policy", "article", "documentation",
    "refill", "delivery", "equipment", "supply", "allowance",
    "continuous", "adjunctive", "blood", "a4238", "a4239", "a4253",
    "a4259", "a4258", "a4257", "a4256", "a4271", "a4250",
    "a4244", "a4245", "a9270", "a9275",
    "e2102", "e2103", "e2100", "e2101", "e2104", "e0607", "e0620",
    "face-to-face", "telehealth",
    "practitioner", "hypoglycemia", "glycemic", "sensor", "transmitter",
    "strip", "medical", "health", "patient", "clinical",
    "dexterity", "impairment", "visual", "acuity", "spring",
    "cartridge", "battery", "receiver", "replacement", "repair",
    "dispensing", "utilization", "denied", "non-covered",
    "maintenance", "routine", "laser", "piercing", "device",
    "powered", "disposable",
]

OFF_TOPIC_SIGNALS = [
    "python", "javascript", "java ", "calculator", "recipe",
    "weather", "movie", "song", "game", "sport", "stock",
    "write me", "generate code", "homework", "essay",
    "translate", "summarize this text", "capital of",
    "cook", "restaurant", "travel", "vacation",
    "who is the president", "tell me a joke", "write a poem",
]


def validate_input(query: str) -> tuple[bool, str]:
    if not query or len(query.strip()) < 3:
        return False, "Query is too short. Please ask a complete question."
    if len(query) > 1000:
        return False, "Query exceeds 1000 characters. Please shorten it."
    for pat in INJECTION_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            return False, "Query contains disallowed patterns. Please rephrase."
    return True, ""


def check_topic_relevance(query: str) -> tuple[bool, float]:
    q = query.lower()
    if any(sig in q for sig in OFF_TOPIC_SIGNALS):
        return False, 0.0
    hits = sum(1 for kw in TOPIC_KEYWORDS if kw in q)
    score = min(hits / 3.0, 1.0)
    return (score >= 0.33 or len(query.split()) <= 5), score


def assess_confidence(scores: list[float]) -> dict:
    if not scores:
        return {"level": "low", "score": 0, "pct": "0%", "color": "red"}
    top = max(scores)
    avg = sum(scores) / len(scores)
    combined = 0.6 * top + 0.4 * avg
    # Normalize cross-encoder scores (typically -12 to +12) to 0-100%
    pct = max(0, min(100, int((combined + 5) / 15 * 100)))
    if pct >= 60:
        return {"level": "high", "score": pct, "pct": f"{pct}%", "color": "green"}
    if pct >= 35:
        return {"level": "medium", "score": pct, "pct": f"{pct}%", "color": "orange"}
    return {"level": "low", "score": pct, "pct": f"{pct}%", "color": "red"}


# ═══════════════════════════════════════════════════════════════════════════
# 2. SECTION-AWARE CHUNKING
# ═══════════════════════════════════════════════════════════════════════════
SECTION_HEADERS = [
    r"^(HOME BLOOD GLUCOSE MONITORS.*)",
    r"^(CONTINUOUS GLUCOSE MONITORS.*)",
    r"^(GENERAL\b.*)",
    r"^(REFILL REQUIREMENTS.*)",
    r"^(REFILL DOCUMENTATION REQUIREMENTS.*)",
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


def _detect_section(line: str) -> str | None:
    s = line.strip()
    for pat in SECTION_HEADERS:
        if re.match(pat, s, re.IGNORECASE):
            return s
    if s.isupper() and 5 < len(s) < 120 and not s[:2] in ("E0", "A4", "A9"):
        return s
    return None


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)


def _split_large(text: str, source: str, section: str) -> list[dict]:
    if len(text) <= CHUNK_SIZE:
        return [{"text": text, "source": source, "section": section}]

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for para in paras:
        if len(cur) + len(para) + 2 <= CHUNK_SIZE:
            cur = f"{cur}\n\n{para}" if cur else para
        else:
            if cur:
                chunks.append({"text": cur, "source": source, "section": section})
            if len(para) > CHUNK_SIZE:
                words = para.split()
                cur = ""
                for w in words:
                    if len(cur) + len(w) + 1 > CHUNK_SIZE:
                        chunks.append({"text": cur.strip(), "source": source, "section": section})
                        cur = ""
                    cur = f"{cur} {w}" if cur else w
            else:
                overlap = cur[-(CHUNK_OVERLAP):] if cur and len(cur) > CHUNK_OVERLAP else ""
                cur = f"{overlap}\n\n{para}" if overlap else para
    if cur.strip():
        chunks.append({"text": cur.strip(), "source": source, "section": section})
    return chunks


def smart_chunk(text: str, source: str) -> list[dict]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    chunks, section, buf = [], "General", ""
    for line in text.split("\n"):
        hdr = _detect_section(line)
        if hdr:
            if buf.strip():
                chunks.extend(_split_large(buf.strip(), source, section))
            section, buf = hdr, line + "\n"
        else:
            buf += line + "\n"
    if buf.strip():
        chunks.extend(_split_large(buf.strip(), source, section))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUERY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
QUERY_TYPES = {
    "coverage": {
        "keywords": ["cover", "eligible", "criteria", "qualify", "approved",
                      "medical necessity", "reasonable and necessary", "denied",
                      "indications", "limitations", "insulin", "non-insulin",
                      "adjunctive", "non-adjunctive", "cgm", "bgm"],
        "label": "Coverage Criteria", "color": "#2563eb",
    },
    "billing": {
        "keywords": ["bill", "code", "hcpcs", "modifier", "claim", "a4238",
                      "a4239", "e2102", "e2103", "strip", "lancet", "supply",
                      "allowance", "quantity", "unit of service", "uos",
                      "cg", "kf", "ks", "kx", "ey"],
        "label": "Billing & Coding", "color": "#d97706",
    },
    "documentation": {
        "keywords": ["document", "swo", "written order", "wopd", "pod",
                      "proof of delivery", "prescription", "signature",
                      "face-to-face", "medical record", "refill", "order",
                      "attestation", "narrative"],
        "label": "Documentation", "color": "#059669",
    },
}


def classify_query(query: str) -> dict:
    q = query.lower()
    scores = {k: sum(kw in q for kw in v["keywords"]) for k, v in QUERY_TYPES.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return {"type": "general", "label": "General", "color": "#6b7280"}
    info = QUERY_TYPES[best]
    return {"type": best, "label": info["label"], "color": info["color"]}


# ═══════════════════════════════════════════════════════════════════════════
# 4. HYBRID RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════
def tokenize(text: str) -> list[str]:
    # Keep alphanumeric codes intact (e.g., A4238, E2102, ICD-10)
    tokens = re.findall(r"[A-Za-z]\d{3,5}|\w+", text.lower())
    return tokens


def build_indices(chunks: list[dict], embed_model: SentenceTransformer):
    texts = [c["text"] for c in chunks]

    embs = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    embs = np.array(embs, dtype="float32")
    faiss.normalize_L2(embs)
    fi = faiss.IndexFlatIP(embs.shape[1])
    fi.add(embs)

    bm = BM25Okapi([tokenize(t) for t in texts])
    return fi, bm


def rrf(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rl in ranked_lists:
        for rank, doc_id in enumerate(rl):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(query: str) -> list[dict]:
    fi = state["faiss_index"]
    bm = state["bm25_index"]
    chunks = state["chunks"]
    em = state["embed_model"]
    ce = state["cross_encoder"]

    q_emb = em.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    _, faiss_ids = fi.search(q_emb, SEMANTIC_TOP_K)
    faiss_ranked = [int(i) for i in faiss_ids[0] if 0 <= i < len(chunks)]

    bm25_scores = bm.get_scores(tokenize(query))
    bm25_ranked = list(np.argsort(bm25_scores)[::-1][:BM25_TOP_K])

    fused = rrf([faiss_ranked, bm25_ranked])
    cand_ids = list(dict.fromkeys(did for did, _ in fused[:SEMANTIC_TOP_K + BM25_TOP_K]))

    if not cand_ids:
        return []

    pairs = [[query, chunks[i]["text"]] for i in cand_ids]
    ce_scores = ce.predict(pairs)

    scored = sorted(zip(cand_ids, ce_scores), key=lambda x: x[1], reverse=True)
    return [{**chunks[did], "score": round(float(s), 4)} for did, s in scored[:RERANK_TOP_K]]


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONVERSATION CONTEXT
# ═══════════════════════════════════════════════════════════════════════════
def expand_follow_up(query: str, history: list[dict]) -> str:
    signals = ["what about", "how about", "and for", "what if",
               "can you also", "tell me more", "elaborate", "similarly"]
    short = len(query.split()) <= 6
    has_signal = any(s in query.lower() for s in signals)
    has_pronoun = bool(re.search(r"\b(it|its|they|them|their|this|that|these|those)\b", query, re.I))

    if (short or has_signal or has_pronoun) and history:
        for m in reversed(history):
            if m.get("role") == "user":
                return f"{m['content']} — follow-up: {query}"
    return query


def build_conv_context(history: list[dict], max_turns: int = 3) -> str:
    recent, turns = [], 0
    for m in reversed(history):
        if turns >= max_turns:
            break
        recent.insert(0, m)
        if m.get("role") == "user":
            turns += 1
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in recent
    ) if recent else ""


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM GENERATION
# ═══════════════════════════════════════════════════════════════════════════
BASE_PROMPT = """You are a Medicare coverage policy expert assistant specializing in Durable Medical Equipment (DME), specifically glucose monitors (BGM and CGM).

STRICT RULES:
1. Answer ONLY from the provided policy excerpts. NEVER use outside knowledge.
2. If the excerpts do not contain the answer, state: "This information is not available in the loaded policy documents."
3. Cite the source document name for every factual claim (e.g., "According to LCD L33822..." or "Per Policy Article A52464...").
4. ALWAYS include exact HCPCS codes (e.g., E2102, A4238), ICD-10 codes, modifiers (CG, KF, KS, KX, EY), specific quantities, and timeframes when they appear in the source text.
5. When listing criteria, include ALL criteria with their original numbering — do not summarize or skip any.
6. Use exact numbers and units from the policy (e.g., "100 test strips", "300 lancets", "3 months", "6 months", "7 years", "30 calendar days", "10 calendar days").
7. Do NOT provide medical advice — only explain what the policy documents state.
8. Structure responses with clear headings, numbered lists, and bullet points.
9. When discussing denial conditions, explicitly state what will be "denied as not reasonable and necessary."
10. Distinguish between BGM and CGM rules when both are relevant."""

TYPE_INSTRUCTIONS = {
    "coverage": "\nFocus on: ALL coverage criteria numbered exactly as in the policy, initial vs continued coverage differences, specific HCPCS codes for each device type, BGM vs CGM differences, and what causes denial. Include ALL sub-criteria (e.g., 4A, 4B).",
    "billing": "\nFocus on: exact HCPCS codes with their full descriptions, ALL required modifiers (CG/KF/KS/KX/EY) and exactly when each applies, quantities per billing period, units of service definitions, bundling/unbundling rules, and Column I/Column II relationships.",
    "documentation": "\nFocus on: ALL required elements listed in the policy, exact timeframes (e.g., 6 months, 7 years, 30 days, 10 days, 12 months), differences between SWO/WOPD/POD, initial vs refill documentation, all delivery methods, and signature requirements.",
    "general": "\nProvide a comprehensive answer covering all relevant aspects. Include specific codes, quantities, timeframes, and regulatory references (e.g., CFR citations, Social Security Act sections) when available.",
}


def generate(query: str, sources: list[dict], api_key: str,
             query_type: str, conv_ctx: str) -> str:
    client = genai.Client(api_key=api_key)

    ctx_block = "\n\n---\n\n".join(
        f"[Source: {s['source']} | Section: {s.get('section', 'N/A')}]\n{s['text']}"
        for s in sources
    )
    type_instr = TYPE_INSTRUCTIONS.get(query_type, "")
    conv_block = f"\nRECENT CONVERSATION:\n{conv_ctx}\n" if conv_ctx else ""

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"POLICY EXCERPTS:\n{ctx_block}\n{conv_block}\nQUESTION: {query}",
        config=genai_types.GenerateContentConfig(
            system_instruction=f"{BASE_PROMPT}{type_instr}",
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    return resp.text


# ═══════════════════════════════════════════════════════════════════════════
# 7. FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    state["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
    print("Loading cross-encoder...")
    state["cross_encoder"] = CrossEncoder(CROSS_ENCODER_NAME)

    print("Processing PDFs and building indices...")
    all_chunks = []
    docs_loaded = 0
    for fname, display in PDF_FILES.items():
        path = PDF_DIR / fname
        if not path.exists():
            print(f"  SKIP  {fname}")
            continue
        text = extract_pdf_text(path)
        all_chunks.extend(smart_chunk(text, display))
        docs_loaded += 1
        print(f"  OK    {display}")

    state["chunks"] = all_chunks
    state["docs_loaded"] = docs_loaded

    fi, bm = build_indices(all_chunks, state["embed_model"])
    state["faiss_index"] = fi
    state["bm25_index"] = bm

    print(f"Ready — {len(all_chunks)} chunks from {docs_loaded} documents")
    yield
    state.clear()


app = FastAPI(title="Medicare DME Policy Assistant", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    query: str
    api_key: str
    history: list[dict] = []


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ready" if state.get("chunks") else "loading",
        "documents_loaded": state.get("docs_loaded", 0),
        "total_chunks": len(state.get("chunks", [])),
        "has_env_key": bool(os.getenv("GEMINI_API_KEY")),
        "models": {
            "embeddings": EMBED_MODEL_NAME,
            "reranker": CROSS_ENCODER_NAME,
            "llm": GEMINI_MODEL,
        },
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    t_start = time.time()

    # Resolve API key: request > environment
    api_key = req.api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"error": "No API key provided. Set it in the sidebar or in the .env file.", "guardrail": "auth"}

    # Guardrail: input validation
    ok, msg = validate_input(req.query)
    if not ok:
        return {"error": msg, "guardrail": "input_validation"}

    relevant, rel_score = check_topic_relevance(req.query)
    if not relevant:
        return {
            "error": "This question doesn't appear related to Medicare DME coverage for glucose monitors. Please rephrase.",
            "guardrail": "topic_scope",
        }

    # Classify
    qtype = classify_query(req.query)

    # Expand follow-ups
    retrieval_query = expand_follow_up(req.query, req.history)

    # Retrieve
    t_ret = time.time()
    sources = hybrid_retrieve(retrieval_query)
    retrieval_s = round(time.time() - t_ret, 3)

    # Confidence
    confidence = assess_confidence([s["score"] for s in sources])

    # Conversation context
    conv_ctx = build_conv_context(req.history)

    # Generate
    t_gen = time.time()
    try:
        answer = generate(req.query, sources, api_key, qtype["type"], conv_ctx)
    except Exception as exc:
        return {"error": f"LLM error: {exc}", "guardrail": "generation"}
    generation_s = round(time.time() - t_gen, 3)

    return {
        "answer": answer,
        "sources": sources,
        "query_type": qtype,
        "confidence": confidence,
        "timing": {
            "retrieval_s": retrieval_s,
            "generation_s": generation_s,
            "total_s": round(time.time() - t_start, 3),
        },
        "error": None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
