"""
Generate a lesson plan using retrieved context + rubric as system prompt.
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from hybrid_retriever import HybridRetriever

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Placeholder rubric summary. Replace with the full rubric V3.2 text later.
RUBRIC_SYSTEM_PROMPT = """You are an assistant that helps generate project-based learning (PBL) lesson plans
for teachers, using established PBL frameworks and official policy documents as reference material.

Follow these core principles at all times, aligned with Gold Standard PBL:
1. Challenging Problem or Question: the project should be framed by an open-ended driving question.
2. Sustained Inquiry: students should engage in an active, in-depth process over time.
3. Authenticity: the project should have real-world context, tools, and impact.
4. Student Voice and Choice: allow students meaningful choices in content, process, or product.
5. Reflection and Critique & Revision: build in structured opportunities for feedback and revision.
6. Public Product: the project should culminate in work shared beyond the classroom.

Use only the provided reference material below. Do not invent facts not supported by it.
If the reference material is insufficient for a specific claim, state that clearly rather than fabricating.
Respond in the same language as the user's request (English or Chinese).
"""

def build_context(retrieved_chunks):
    lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        lines.append(f"[Source {i}: {chunk['source']}] {chunk['text']}")
    return "\n\n".join(lines)

def generate_lesson_plan(user_prompt, retriever, top_k=5, detail_level="detailed"):
    retrieved = retriever.retrieve(user_prompt, top_k=top_k)
    context = build_context(retrieved)

    if detail_level == "concise":
        length_instruction = "\nRespond concisely. Limit your answer to the most essential points only, avoiding elaboration or extended explanation of each point."
        max_tok = 800
    else:
        length_instruction = ""
        max_tok = 4000

    full_user_message = f"""Reference material:
{context}

Task: {user_prompt}{length_instruction}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tok,
        system=RUBRIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": full_user_message}],
    )

    return {
        "output": response.content[0].text,
        "retrieved_sources": [
            {
                "id": c["id"],
                "source": c["source"],
                "score": c["combined_score"],  # HybridRetriever returns combined_score instead of score
                "text": c["text"],
            }
            for c in retrieved
        ],
    }

if __name__ == "__main__":
    # dense_weight=1.0 for Chinese queries (BM25 provides no net benefit for
    # same-language Chinese matching given bge-m3's strong multilingual
    # semantics — confirmed via Part A/B cross-validated grid search, see
    # Hybrid_Retrieval_Quantitative_Evaluation_Report.md Section 15).
    # Non-Chinese queries remain hardcoded to 0.9 via query routing
    # (see hybrid_retriever.py _is_likely_chinese()).
    retriever = HybridRetriever(corpus_path="../../data/processed/chunks.jsonl", dense_weight=1.0)

    test_prompts = [
        "Design a project-based learning unit for a Grade 6 mathematics class using a driving question, with a public product and a plan for student reflection.",
        "为初中科学课设计一个跨学科项目式学习方案，包含驱动性问题和评价量规。",
        "How should I structure a rubric to assess student collaboration during a project?",
    ]

    for i, prompt in enumerate(test_prompts, start=1):
        print("\n" + "#"*70)
        print(f"# TEST {i}: {prompt}")
        print("#"*70)

        result = generate_lesson_plan(prompt, retriever)

        print("\n--- RETRIEVED SOURCES ---")
        for s in result["retrieved_sources"]:
            print(f"  {s['id']} ({s['source']}) - score: {s['score']:.4f}")

        print("\n--- GENERATED LESSON PLAN ---")
        print(result["output"])