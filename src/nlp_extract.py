from __future__ import annotations
import re
from typing import List, Dict, Set
import spacy
from rapidfuzz import fuzz

_NLP = None

# A small curated skill list (you can expand anytime)
SKILL_BANK = {
    "python", "java", "javascript", "typescript", "sql",
    "spring", "spring boot", "microservices",
    "aws", "docker", "kubernetes", "terraform",
    "langchain", "openai", "llm", "rag", "chroma", "vector database",
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow",
    "rest", "graphql", "system design",
}

def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP

def extract_entities(text: str) -> Dict[str, List[str]]:
    nlp = _get_nlp()
    doc = nlp(text)
    ents = {}
    for ent in doc.ents:
        ents.setdefault(ent.label_, set()).add(ent.text.strip())
    return {k: sorted(list(v)) for k, v in ents.items()}

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\+\#\.\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_skills(text: str, threshold: int = 90) -> List[str]:
    """
    Simple NLP skill extraction:
    - normalize text
    - fuzzy match skill bank terms against text windows
    """
    t = _normalize(text)
    found: Set[str] = set()

    # Fast contains checks for multiword skills
    for skill in SKILL_BANK:
        sk = _normalize(skill)
        if sk in t:
            found.add(skill)

    # Fuzzy match for near-misses (small NLP flavor)
    # Example: "springboot" still matches "spring boot"
    words = t.split()
    for skill in SKILL_BANK:
        sk = _normalize(skill)
        sk_words = sk.split()
        n = len(sk_words)
        if n <= 1:
            continue
        for i in range(0, max(0, len(words) - n + 1)):
            window = " ".join(words[i:i+n])
            if fuzz.ratio(window, sk) >= threshold:
                found.add(skill)

    return sorted(found)

def estimate_years_experience(text: str) -> int:
    """
    Very simple heuristic: look for patterns like "X years"
    """
    t = text.lower()
    matches = re.findall(r"(\d+)\s*\+?\s*years", t)
    if not matches:
        return 0
    years = [int(m) for m in matches]
    return max(years) if years else 0
