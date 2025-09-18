# tools.py
import re
from typing import Dict, Any, Optional, List

_STOP = {
    "the","a","an","of","to","in","and","or","for","on","by","with","is","are","was","were","be","as",
    "at","from","that","this","it","its","into","than","then","there","their","them","these","those",
    "about","also","can","could","should","will","would","may","might","must"
}

def _tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-']+", s.lower())

def _key_terms(s: str) -> List[str]:
    return [t for t in _tokens(s) if t not in _STOP and len(t) > 2]

def _chunk(s: str, size: int) -> List[str]:
    return [s[i:i+size] for i in range(0, len(s), size)]

def _score(ch: str, keys: set[str]) -> int:
    toks = set(_tokens(ch))
    return len(keys & toks)

def compress_context_for_question(
    context: str,
    question: str,
    choices: Dict[str, str],
    chunk_size: int = 1000,
    max_chars: int = 450000,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Rank context chunks by keyword overlap with {question+choices} and keep the best ones.
    Returns {'compressed': str, 'kept_indices': List[int], 'original_len': int}
    """
    # Normalize choices mapping
    norm = {k: choices.get(k) or choices.get(f"choice_{k}") or "" for k in ("A","B","C","D")}
    keys = set(_key_terms(question))
    for k in ("A","B","C","D"):
        keys.update(_key_terms(norm[k]))

    chunks = _chunk(context, chunk_size)
    scored = [(i, ch, _score(ch, keys)) for i, ch in enumerate(chunks)]
    scored.sort(key=lambda x: x[2], reverse=True)

    if top_k is None:
        top_k = max(1, max_chars // max(1, chunk_size))

    selected = sorted(scored[:top_k], key=lambda x: x[0])
    compressed = "".join(ch for _, ch, _ in selected)
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars]
    if len(context) < max_chars:
        compressed = context
    
    return {
        "compressed": compressed,
        "kept_indices": [i for i, _, _ in selected],
        "original_len": len(context),
    }

def strict_answer_letter(text: str) -> Optional[str]:
    """Extract a clean leading letter A|B|C|D. Returns None if not found."""
    m = re.match(r'^\s*([ABCD])\b', text.strip().upper())
    return m.group(1) if m else None
