from __future__ import annotations
import re
import json
from urllib.parse import urlparse
from src.models.llm import LLMClient

_FAITHFULNESS_SYS = (
    "You are a strict factual-consistency evaluator for a RAG system. You are "
    "given CONTEXT passages and an ANSWER. Estimate what fraction of the answer's "
    "factual claims are directly supported by the context.\n\n"
    "Rules:\n"
    "- Break the answer into atomic factual claims. Ignore opinions, hedges, and "
    "meta statements such as 'the sources do not cover X'.\n"
    "- A claim counts as supported ONLY if it can be verified from the context "
    "alone. Do not use outside knowledge. Citation markers like [1] are NOT "
    "evidence — check the actual passage text.\n"
    "- score = supported_claims / total_claims, a float in [0, 1]. If the answer "
    "makes no factual claims, score = 1.0.\n"
    "- List each unsupported claim briefly (a short paraphrase is fine).\n\n"
    "Return ONLY valid JSON, no prose, no markdown:\n"
    '{"score": <float 0..1>, "unsupported_claims": ["...", "..."]}'
)

_RELEVANCE_SYS = (
    "You judge whether an ANSWER addresses the QUESTION that was asked. You are "
    "judging relevance and completeness, NOT factual accuracy.\n\n"
    "Rules:\n"
    "- 1.0 = directly and fully addresses the question; 0.0 = off-topic or "
    "evasive; use the range in between for partial answers.\n"
    "- An honest 'the retrieved sources do not cover this' for a genuinely "
    "unanswerable question still addresses the question — score it high.\n\n"
    "Return ONLY valid JSON, no prose, no markdown:\n"
    '{"score": <float 0..1>, "reasoning": "<one sentence>"}'
)

def _norm_domain(url: str) -> str:
    domain=urlparse(url).netloc.lower()
    return domain.removeprefix("www."); #strip leading "www."



def citation_accuracy(report_sources: list[str], must_cite: list[str]) -> float:
    if not must_cite:
      return 1.0
    got = set(_norm_domain(url) for url in report_sources)
    want = set(_norm_domain(url) for url in must_cite)
    intersection=got & want
    return  len(intersection)/len(want)

 


async def faithfulness(answer: str, context: str, judge: LLMClient) -> tuple[float, list[str]]:
    if not answer or not context:
        return (0.0, [])
    prompt = f"Context: {context}\n\nAnswer: {answer}"
    
    try:
        # Await the async network call
        response_text = await judge.generate_json(
            prompt=prompt, 
            system=_FAITHFULNESS_SYS, 
            temperature=0.0
        )
        # Parse the JSON string
        result = json.loads(response_text)
        
        # Safely extract the values 
        score = float(result.get("score", 0.0))
        unsupported_claims = result.get("unsupported_claims", [])
        return (score, unsupported_claims)
        
    except Exception as e:
        # Handle the "on error" fallback
        print(f"Faithfulness parsing error: {e}")
        return (0.0, [])


async def answer_relevance(question: str, answer: str, judge: LLMClient) -> float:
    if not answer:
        return 0.0
    prompt = f"Question: {question}\n\nAnswer: {answer}"
    try:
        # Await the async network call
        response_text = await judge.generate_json(
            prompt=prompt, 
            system=_RELEVANCE_SYS, 
            temperature=0.0
        )
        # Parse the JSON string
        result = json.loads(response_text)
        
        #Safely extract the values 
        score = float(result.get("score", 0.0))
        
        return score
        
    except Exception as e:
        # 5. Handle the "on error" fallback
        print(f"answer_relevance error: {e}")
        return 0.0


def extract_cited_markers(report_text: str) -> list[int]:
    
    #Extracts numerical citation markers like [1], [2] from a text.
    #Returns a deduplicated and sorted list of the integers.
    # Find all digits enclosed in square brackets
    matches = re.findall(r"\[(\d+)\]", report_text)
    
    # Convert matches to integers, deduplicate with set(), and sort
    return sorted(list(set(int(match) for match in matches)))