"""
Automated test runner for Medicare DME Policy Assistant.
Runs all 50 questions from test_questions.json against the local API,
evaluates answer accuracy via keyword matching, and checks citation accuracy.

Usage:
    python run_tests.py            # run all questions
    python run_tests.py --ids 1 5  # run specific question IDs
"""

import json
import time
import argparse
import re
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8000/api/chat"
HEALTH_URL = "http://localhost:8000/api/health"
TEST_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_FILE = Path(__file__).parent / "test_results.json"

# Keywords that MUST appear in the answer for it to be considered accurate
# Mapped by question ID
CRITICAL_KEYWORDS = {
    1: ["diabetes", "training", "prescription"],
    2: ["denied", "not reasonable and necessary"],
    3: ["E2100", "E2101", "voice synthesizer", "visual impairment", "20/200"],
    4: ["E2101", "manual dexterity", "not dependent"],
    5: ["100 test strips", "100 lancets", "3 months"],
    6: ["300 test strips", "300 lancets", "3 months"],
    7: ["100 test strips", "100 lancets", "6 months", "telehealth"],
    8: ["300 test strips", "300 lancets", "6 months"],
    9: ["diabetes", "training", "FDA", "glycemic", "insulin"],
    10: ["54mg", "level 2", "level 3", "hypoglycemi"],
    11: ["6 months", "in-person", "adherence"],
    12: ["adjunctive", "non-adjunctive", "stand-alone", "BGM"],
    13: ["E2102", "E2103"],
    14: ["A4238", "A4239", "3", "90"],
    15: ["non-adjunctive", "denied", "replace"],
    16: ["adjunctive", "A4238", "separately"],
    17: ["KX", "KS", "CG", "KF"],
    18: ["CG", "criteria"],
    19: ["KF", "Class III", "E2102", "A4238"],
    20: ["name", "date", "description", "signature", "NPI"],
    21: ["new order", "purchase", "change", "replaced"],
    22: ["WOPD", "before delivery", "6 months"],
    23: ["direct delivery", "shipping", "nursing facility"],
    24: ["name", "address", "description", "quantity", "date", "signature"],
    25: ["7 years"],
    26: ["30 calendar days", "10 calendar days", "affirmative"],
    27: ["name", "description", "affirmative", "date"],
    28: ["A4238", "A4239", "do not apply"],
    29: ["A4259", "A4253", "lancet", "strip"],
    30: ["A4258", "6 months", "one"],
    31: ["E0620", "laser", "denied", "not established"],
    32: ["A4253", "50", "A4259", "100"],
    33: ["A4271", "cartridge", "50"],
    34: ["three", "3", "month"],
    35: ["A9270", "non-covered"],
    36: ["A9275", "non-covered", "DME"],
    37: ["A4244", "A4245", "non-covered"],
    38: ["A4250", "non-covered"],
    39: ["office records", "hospital", "medical records"],
    40: ["supplier", "attestation", "not", "sufficient"],
    41: ["face-to-face", "6 months", "410.38"],
    42: ["does not need to be", "prescriber", "verify"],
    43: ["loss", "irreparable", "5 years", "RUL"],
    44: ["routine", "maintenance", "not covered"],
    45: ["reasonable and necessary", "1862"],
    46: ["benefit category", "reasonable and necessary", "regulatory"],
    47: ["SWO", "before", "denied"],
    48: ["E2103", "E0607", "A4253"],
    49: ["12 months", "medical record", "refill"],
    50: [],  # Guardrail test — should be rejected
}

# Expected source documents for citation accuracy
EXPECTED_SOURCES = {
    1: ["LCD L33822"],
    2: ["LCD L33822"],
    3: ["LCD L33822"],
    4: ["LCD L33822"],
    5: ["LCD L33822"],
    6: ["LCD L33822"],
    7: ["LCD L33822"],
    8: ["LCD L33822"],
    9: ["LCD L33822"],
    10: ["LCD L33822"],
    11: ["LCD L33822"],
    12: ["LCD L33822"],
    13: ["LCD L33822"],
    14: ["LCD L33822", "A52464"],
    15: ["A52464"],
    16: ["A52464"],
    17: ["A52464"],
    18: ["A52464"],
    19: ["A52464"],
    20: ["A55426"],
    21: ["A55426"],
    22: ["A55426"],
    23: ["A55426"],
    24: ["A55426"],
    25: ["A55426"],
    26: ["LCD L33822", "A55426"],
    27: ["A55426"],
    28: ["LCD L33822", "A52464"],
    29: ["LCD L33822"],
    30: ["LCD L33822"],
    31: ["LCD L33822"],
    32: ["A52464"],
    33: ["A52464"],
    34: ["LCD L33822"],
    35: ["A52464"],
    36: ["A52464"],
    37: ["A52464"],
    38: ["A52464"],
    39: ["A55426"],
    40: ["A55426"],
    41: ["A55426"],
    42: ["A55426"],
    43: ["A55426"],
    44: ["A55426"],
    45: ["LCD L33822"],
    46: ["LCD L33822", "A52464"],
    47: ["LCD L33822"],
    48: ["A52464"],
    49: ["A55426"],
    50: [],
}


def check_server():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        data = r.json()
        if data.get("status") != "ready":
            print("Server is still loading. Wait for startup to complete.")
            sys.exit(1)
        print(f"Server ready: {data['total_chunks']} chunks from {data['documents_loaded']} docs\n")
    except requests.ConnectionError:
        print("Cannot connect to server at localhost:8000. Start it with: python main.py")
        sys.exit(1)


SYNONYMS = {
    "diabetes": ["diabetes", "diabetic", "diabetes mellitus", "dm"],
    "training": ["training", "trained", "education", "educated", "instructed", "sufficient training"],
    "prescription": ["prescription", "prescribed", "order", "ordering"],
    "denied": ["denied", "denial", "deny", "will be denied", "shall be denied"],
    "not reasonable and necessary": ["not reasonable and necessary", "denied as not reasonable", "not r&n"],
    "voice synthesizer": ["voice synthesizer", "voice", "synthesizer", "integrated voice", "sound output"],
    "visual impairment": ["visual impairment", "visual acuity", "visually impaired", "severe visual"],
    "20/200": ["20/200", "20 200"],
    "6 months": ["6 months", "six months", "six (6) months", "6-month", "every six months"],
    "3 months": ["3 months", "three months", "three (3) months", "3-month", "every 3 months", "every three months"],
    "telehealth": ["telehealth", "tele-health", "telemedicine", "medicare-approved telehealth"],
    "in-person": ["in-person", "in person", "face-to-face", "office visit"],
    "adherence": ["adherence", "adherent", "adhere", "compliance", "compliant", "document adherence"],
    "stand-alone": ["stand-alone", "standalone", "stand alone", "without the need for"],
    "insulin": ["insulin", "insulin-treated", "insulin treated", "insulin administrations"],
    "FDA": ["fda", "food and drug administration", "fda indications"],
    "glycemic": ["glycemic", "glycemic control", "glucose control", "blood sugar"],
    "replace": ["replace", "replacement", "replacing", "replaces"],
    "separately": ["separately", "separate", "in addition to", "billed separately"],
    "do not apply": ["do not apply", "does not apply", "not applicable", "are not applicable"],
    "not covered": ["not covered", "non-covered", "noncovered", "denied", "will not be covered"],
    "not established": ["not established", "has not been established", "not been established"],
    "office records": ["office records", "office record", "practitioner's office", "medical record", "treating practitioner"],
    "hospital": ["hospital", "hospital records", "inpatient"],
    "sufficient": ["sufficient", "not sufficient", "insufficient", "do not provide sufficient"],
    "attestation": ["attestation", "attest", "prepared statements"],
    "signature": ["signature", "signed", "signing", "practitioner's signature"],
    "NPI": ["npi", "national provider identifier", "provider identifier"],
    "direct delivery": ["direct delivery", "directly to the beneficiary", "deliver directly"],
    "shipping": ["shipping", "shipping service", "delivery service", "mail order", "via shipping"],
    "nursing facility": ["nursing facility", "nursing home", "nursing facilities"],
    "name": ["name", "beneficiary's name", "beneficiary name"],
    "address": ["address", "delivery address"],
    "description": ["description", "item description", "description of the item", "general description"],
    "quantity": ["quantity", "quantities", "quantity delivered", "quantity dispensed"],
    "date": ["date", "order date", "date delivered", "date of delivery", "date of refill"],
    "affirmative": ["affirmative", "affirmative response", "affirms", "confirms", "confirmation"],
    "10 calendar days": ["10 calendar days", "ten calendar days", "ten (10) calendar days", "10 days prior"],
    "30 calendar days": ["30 calendar days", "thirty calendar days", "thirty (30) calendar days", "30 days prior"],
    "7 years": ["7 years", "seven years", "seven (7) years"],
    "12 months": ["12 months", "twelve months", "twelve (12) months", "preceding 12 months"],
    "refill": ["refill", "refills", "replenish", "refill request"],
    "100 test strips": ["100 test strips", "100 strips", "one hundred test strips", "up to 100 test strips"],
    "100 lancets": ["100 lancets", "one hundred lancets", "up to 100 lancets"],
    "300 test strips": ["300 test strips", "300 strips", "three hundred test strips", "up to 300 test strips"],
    "300 lancets": ["300 lancets", "three hundred lancets", "up to 300 lancets"],
    "5 years": ["5 years", "five years", "five (5) years", "less than 5"],
    "RUL": ["rul", "reasonable useful lifetime", "useful lifetime"],
    "410.38": ["410.38", "42 cfr 410", "cfr 410.38"],
    "1862": ["1862", "section 1862", "social security act"],
    "reasonable and necessary": ["reasonable and necessary", "r&n", "reasonable & necessary"],
    "benefit category": ["benefit category", "medicare benefit", "defined medicare benefit"],
    "regulatory": ["regulatory", "statutory", "regulatory requirements", "statutory and regulatory"],
    "maintenance": ["maintenance", "periodic maintenance", "routine periodic maintenance"],
}


def flexible_match(keyword: str, answer: str) -> bool:
    answer_lower = answer.lower()
    if keyword.lower() in answer_lower:
        return True
    synonyms = SYNONYMS.get(keyword.lower(), SYNONYMS.get(keyword, []))
    return any(syn.lower() in answer_lower for syn in synonyms)


def evaluate_answer(qid: int, answer: str, sources: list, is_guardrail: bool) -> dict:
    answer_lower = answer.lower()

    if is_guardrail:
        rejected = any(s in answer_lower for s in [
            "not available", "not related", "rephrase", "don't appear",
            "doesn't appear", "cannot help", "outside", "not relevant",
        ])
        return {
            "keyword_hits": 0, "keyword_total": 0,
            "keyword_pct": 100.0 if rejected else 0.0, "keyword_pass": rejected,
            "citation_hits": 0, "citation_total": 0,
            "citation_pct": 100.0 if rejected else 0.0, "citation_pass": rejected,
            "matched_keywords": [], "missed_keywords": [],
            "matched_sources": [], "missed_sources": [],
        }

    keywords = CRITICAL_KEYWORDS.get(qid, [])
    matched = [kw for kw in keywords if flexible_match(kw, answer)]
    missed = [kw for kw in keywords if not flexible_match(kw, answer)]
    kw_pct = (len(matched) / len(keywords) * 100) if keywords else 100.0

    expected = EXPECTED_SOURCES.get(qid, [])
    source_names = " ".join(s.get("source", "") for s in (sources or [])).lower()
    answer_combined = (answer_lower + " " + source_names)
    matched_src = [s for s in expected if s.lower() in answer_combined]
    missed_src = [s for s in expected if s.lower() not in answer_combined]
    cite_pct = (len(matched_src) / len(expected) * 100) if expected else 100.0

    return {
        "keyword_hits": len(matched), "keyword_total": len(keywords),
        "keyword_pct": round(kw_pct, 1), "keyword_pass": kw_pct >= 60,
        "citation_hits": len(matched_src), "citation_total": len(expected),
        "citation_pct": round(cite_pct, 1), "citation_pass": cite_pct >= 50,
        "matched_keywords": matched, "missed_keywords": missed,
        "matched_sources": matched_src, "missed_sources": missed_src,
    }


def run_tests(question_ids: list[int] | None = None):
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]

    total = len(questions)
    results = []
    passed_kw = 0
    passed_cite = 0
    total_kw_pct = 0
    total_cite_pct = 0
    errors = 0

    print(f"Running {total} questions...\n")
    print(f"{'#':<4} {'Category':<22} {'KW%':<7} {'Cite%':<7} {'Conf':<7} {'Time':<7} {'Status'}")
    print("-" * 85)

    for q in questions:
        qid = q["id"]
        is_guardrail = q["category"] == "Guardrail Test"

        try:
            t0 = time.time()
            resp = requests.post(API_URL, json={
                "query": q["question"],
                "api_key": "",
                "history": [],
            }, timeout=60)
            elapsed = round(time.time() - t0, 1)
            data = resp.json()

            if data.get("error") and not is_guardrail:
                errors += 1
                status = "ERROR"
                ev = {"keyword_pct": 0, "citation_pct": 0, "keyword_pass": False, "citation_pass": False}
                answer = data["error"]
                confidence = "N/A"
            elif data.get("error") and is_guardrail:
                ev = evaluate_answer(qid, data["error"], [], True)
                answer = data["error"]
                confidence = "N/A"
                status = "PASS" if ev["keyword_pass"] else "FAIL"
            else:
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                conf_data = data.get("confidence", {})
                confidence = conf_data.get("pct", "?")
                ev = evaluate_answer(qid, answer, sources, is_guardrail)
                status = "PASS" if (ev["keyword_pass"] and ev["citation_pass"]) else "FAIL"

            if ev["keyword_pass"]:
                passed_kw += 1
            if ev["citation_pass"]:
                passed_cite += 1
            total_kw_pct += ev["keyword_pct"]
            total_cite_pct += ev["citation_pct"]

            color = "\033[92m" if status == "PASS" else "\033[91m" if status == "FAIL" else "\033[93m"
            reset = "\033[0m"

            print(f"{qid:<4} {q['category']:<22} {ev['keyword_pct']:<7.1f} {ev['citation_pct']:<7.1f} {str(confidence):<7} {elapsed:<7.1f} {color}{status}{reset}")

            if ev.get("missed_keywords"):
                print(f"     Missing keywords: {', '.join(ev['missed_keywords'])}")

            results.append({
                "id": qid,
                "category": q["category"],
                "question": q["question"],
                "expected": q["expected_answer"],
                "actual_answer": answer[:500],
                "confidence": confidence,
                "elapsed_s": elapsed,
                "evaluation": ev,
                "status": status,
            })

            # Rate limit: delay between requests to avoid Groq TPM/TPD limits
            time.sleep(2)

        except Exception as exc:
            errors += 1
            print(f"{qid:<4} {q['category']:<22} {'—':<7} {'—':<7} {'—':<7} {'—':<7} \033[93mERROR: {exc}\033[0m")
            results.append({
                "id": qid,
                "category": q["category"],
                "question": q["question"],
                "status": "ERROR",
                "error": str(exc),
            })

    # Summary
    print("\n" + "=" * 85)
    print(f"\n  RESULTS SUMMARY")
    print(f"  {'-' * 40}")
    print(f"  Total questions:      {total}")
    print(f"  Keyword accuracy:     {passed_kw}/{total} passed ({total_kw_pct / total:.1f}% avg)")
    print(f"  Citation accuracy:    {passed_cite}/{total} passed ({total_cite_pct / total:.1f}% avg)")
    print(f"  Overall pass rate:    {sum(1 for r in results if r.get('status') == 'PASS')}/{total} ({sum(1 for r in results if r.get('status') == 'PASS') / total * 100:.1f}%)")
    print(f"  Errors:               {errors}")

    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "kw_sum": 0, "cite_sum": 0}
        categories[cat]["total"] += 1
        ev = r.get("evaluation", {})
        if r.get("status") == "PASS":
            categories[cat]["passed"] += 1
        categories[cat]["kw_sum"] += ev.get("keyword_pct", 0)
        categories[cat]["cite_sum"] += ev.get("citation_pct", 0)

    print(f"\n  CATEGORY BREAKDOWN")
    print(f"  {'Category':<22} {'Pass':<8} {'Avg KW%':<10} {'Avg Cite%'}")
    print(f"  {'-' * 55}")
    for cat, data in sorted(categories.items()):
        avg_kw = data["kw_sum"] / data["total"] if data["total"] else 0
        avg_cite = data["cite_sum"] / data["total"] if data["total"] else 0
        print(f"  {cat:<22} {data['passed']}/{data['total']:<5} {avg_kw:<10.1f} {avg_cite:.1f}")

    # Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total": total,
                "passed": sum(1 for r in results if r.get("status") == "PASS"),
                "failed": sum(1 for r in results if r.get("status") == "FAIL"),
                "errors": errors,
                "avg_keyword_accuracy": round(total_kw_pct / total, 1),
                "avg_citation_accuracy": round(total_cite_pct / total, 1),
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Detailed results saved to: {RESULTS_FILE}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Medicare DME Policy Assistant")
    parser.add_argument("--ids", nargs="+", type=int, help="Run specific question IDs")
    args = parser.parse_args()

    check_server()
    run_tests(args.ids)
