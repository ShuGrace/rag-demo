"""
Faithfulness Checker: decomposes generated text into atomic claims and verifies
each claim against the retrieved source passages using an LLM judge.
Produces a per-claim verdict and an overall faithfulness score.

OPTIMIZED VERSION: batches all claim verifications into a SINGLE API call
instead of one call per claim, drastically reducing token usage (source
passages no longer get re-sent once per claim).
"""
import os
import json
import re
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

CLAIM_EXTRACTION_PROMPT = """You will be given a piece of text (a lesson plan or educational document).
Break it down into a list of atomic, independently verifiable factual claims.

Rules:
- Each claim should be a single, self-contained statement.
- Skip subjective statements, instructions to the teacher, or formatting/structural text (e.g. "Lesson 1", table headers).
- Focus on claims that assert something as fact (e.g. "Collaboration rubrics include four performance levels", "The driving question should be open-ended").
- Do NOT include claims that are just creative content invented for this specific lesson (e.g. a specific city name, a specific student example) — only include claims that assert a general principle, framework element, or fact that should be traceable to a source.
- Return ONLY a JSON array of strings, nothing else. Example: ["claim 1 text", "claim 2 text"]
"""

# Batched verification prompt: all claims verified against all sources in ONE call.
BATCH_VERIFICATION_PROMPT = """You are verifying whether each CLAIM below is supported by the given SOURCE PASSAGES.

SOURCE PASSAGES:
{sources}

CLAIMS TO VERIFY (numbered):
{numbered_claims}

For EACH claim, judge whether it is:
- "supported": the source passages contain clear evidence for this claim
- "partially_supported": the source passages contain related but not fully matching evidence
- "unsupported": the source passages do not contain evidence for this claim

Return ONLY a JSON array of verdicts, in the SAME ORDER as the numbered claims, with no other text.
Example format: ["supported", "unsupported", "partially_supported"]
The array must have exactly {n_claims} elements, one per claim, in order.
"""


def extract_claims(text, model="claude-sonnet-4-6"):
    """Use the LLM to decompose text into atomic claims."""
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=CLAIM_EXTRACTION_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = response.content[0].text.strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        claims = json.loads(match.group(0))
        return [c for c in claims if isinstance(c, str) and len(c.strip()) > 5]
    except json.JSONDecodeError:
        return []


def verify_claims_batch(claims, retrieved_sources, model="claude-sonnet-4-6", max_tokens=2000):
    """
    Verify ALL claims against the source passages in a SINGLE API call.
    This is the key token-saving change vs. the original per-claim loop:
    source passages are sent once, not once per claim.
    """
    if not claims:
        return []

    sources_text = "\n\n".join(
        f"[Source: {s['source']}]\n{s['text']}" for s in retrieved_sources
    )
    numbered_claims = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

    prompt = BATCH_VERIFICATION_PROMPT.format(
        sources=sources_text,
        numbered_claims=numbered_claims,
        n_claims=len(claims),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    verdicts = None
    if match:
        try:
            verdicts = json.loads(match.group(0))
        except json.JSONDecodeError:
            verdicts = None

    # Fallback / safety net: if the model didn't return a clean, correctly-sized
    # JSON array, mark everything "unclear" rather than silently misaligning
    # claims with the wrong verdicts.
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        print(f"  [WARNING] Batch verification returned malformed output "
              f"(expected {len(claims)} verdicts, got "
              f"{len(verdicts) if isinstance(verdicts, list) else 'non-list'}). "
              f"Marking all claims 'unclear'. Raw response: {raw[:200]}")
        return ["unclear"] * len(claims)

    normalized = []
    for v in verdicts:
        v = str(v).strip().lower()
        if "unsupported" in v:
            normalized.append("unsupported")
        elif "partially" in v:
            normalized.append("partially_supported")
        elif "supported" in v:
            normalized.append("supported")
        else:
            normalized.append("unclear")
    return normalized


def check_faithfulness(generated_text, retrieved_sources, verbose=True):
    """
    Full pipeline: extract claims from generated_text, verify ALL of them
    against retrieved_sources in a single batched call, and return a
    faithfulness report.
    """
    if verbose:
        print("Extracting claims from generated text...")
    claims = extract_claims(generated_text)
    if verbose:
        print(f"  Found {len(claims)} claims.\n")
        print("Verifying all claims in a single batched API call...")

    if not claims:
        return {"claims": [], "faithfulness_score": None, "n_claims": 0}

    verdicts = verify_claims_batch(claims, retrieved_sources)

    results = []
    score_map = {"supported": 1.0, "partially_supported": 0.7, "unsupported": 0.0, "unclear": 0.0}
    for i, (claim, verdict) in enumerate(zip(claims, verdicts), start=1):
        results.append({"claim": claim, "verdict": verdict})
        if verbose:
            print(f"[{i}/{len(claims)}] [{verdict}] {claim[:100]}")

    total_score = sum(score_map[r["verdict"]] for r in results)
    faithfulness_score = round(total_score / len(results), 4)

    n_supported = sum(1 for r in results if r["verdict"] == "supported")
    n_partial = sum(1 for r in results if r["verdict"] == "partially_supported")
    n_unsupported = sum(1 for r in results if r["verdict"] == "unsupported")

    return {
        "claims": results,
        "faithfulness_score": faithfulness_score,
        "n_claims": len(results),
        "n_supported": n_supported,
        "n_partially_supported": n_partial,
        "n_unsupported": n_unsupported,
    }


if __name__ == "__main__":
    # Quick standalone test using generate.py's pipeline.
    # Updated to use HybridRetriever (matches the now-current production retriever
    # in generate.py) instead of the old InMemoryRetriever.
    from hybrid_retriever import HybridRetriever
    from generate import generate_lesson_plan
    import pathlib

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    retriever = HybridRetriever(
        corpus_path=str(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"),
        dense_weight=0.5,
    )

    test_prompt = "How should I structure a rubric to assess student collaboration during a project?"
    result = generate_lesson_plan(test_prompt, retriever)

    print("="*70)
    print("GENERATED TEXT (preview):")
    print("="*70)
    print(result["output"][:500] + "...\n")

    report = check_faithfulness(result["output"], result["retrieved_sources"])

    print("\n" + "="*70)
    print("FAITHFULNESS REPORT")
    print("="*70)
    print(f"Overall Faithfulness Score: {report['faithfulness_score']}")
    print(f"Total claims: {report['n_claims']}")
    print(f"  Supported: {report['n_supported']}")
    print(f"  Partially supported: {report['n_partially_supported']}")
    print(f"  Unsupported: {report['n_unsupported']}")