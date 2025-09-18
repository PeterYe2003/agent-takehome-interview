#!/usr/bin/env python3
"""
tool_code_evolution.py
Evolve Python functions by mutating their code with Claude.
Evaluates each generation on a benchmark, keeps the best variant.
"""

import os
import re
import inspect
import textwrap
import random
import asyncio
import logging
from typing import Callable, List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tool-code-evolution")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def search_tool(query: str, docs: list[str]) -> str:
    """Return the first document containing the query (case-insensitive)."""
    for d in docs:
        if query.lower() in d.lower():
            return d
    return "Not found"


# ================= Benchmark =================
benchmark: List[Dict[str, Any]] = [
    {
        "query": "apple",
        "docs": ["banana", "orange", "apple pie"],
        "expected": "apple pie",
    },
    {
        "query": "machine learning",
        "docs": ["deep learning basics", "machine learning guide"],
        "expected": "machine learning guide",
    },
    {
        "query": "typo test",
        "docs": ["typo test doc", "typo"],
        "expected": "typ test doc",
    },
]


# ================= Helpers =================
def _strip_md_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:\w+)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
    return code.strip()


def _clean_src(func: Callable) -> str:
    """Dedent source and strip decorators."""
    target = getattr(func, "func", func)
    src = textwrap.dedent(inspect.getsource(target))
    lines = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("@"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def get_callable(tool_obj: Any) -> Callable:
    """
    Return a plain callable for evaluation.
    - If it's a wrapped function, unwrap underlying python function via .func (or common fallbacks).
    - Else assume it's already a function.
    """
    for attr in ("func", "_func", "_callable"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    if callable(tool_obj):
        return tool_obj
    raise TypeError("Provided tool is not callable and cannot be unwrapped.")


def compile_code_to_plain_function(code_str: str) -> Callable:
    """
    Compile mutated code as a PLAIN FUNCTION.
    """
    code = _strip_md_fences(code_str)
    g: Dict[str, Any] = {}
    l: Dict[str, Any] = {}

    exec(code, g, l)

    tools = [v for v in l.values() if callable(v)]
    if not tools:
        raise RuntimeError("No callable found in mutated code.")
    return tools[0]


# ================= Evaluation =================
def evaluate_tool(tool_like: Any, bench: List[Dict[str, Any]]) -> float:
    """
    Evaluate a function on the benchmark.
    """
    fn = get_callable(tool_like)
    correct = 0
    for item in bench:
        try:
            out = fn(item["query"], item["docs"])
            out_s = (out or "").lower()
            if item["expected"].lower() in out_s:
                correct += 1
        except Exception as e:
            logger.warning(f"Tool crashed on query='{item['query']}': {e}")
    return (correct / len(bench)) if bench else 0.0


def mutate_with_claude(func: Any) -> Tuple[Callable, str]:
    src = _clean_src(func)
    prompt = f"""
Improve the following Python function.
Keep the SAME NAME and signature.

Code:

{src}

Requirements:
- Rank docs for best match using token overlap and light typo tolerance (edit distance).
- Deterministic, standard library only.
- Return ONLY valid Python code (no markdown fences).
- DO NOT use any excess text like "Here's an improved version of the search_tool function..."
- Have ONLY the code in your response as if it was in a .py file.
"""
    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    code = "".join(getattr(block, "text", "") for block in (resp.content or []))
    logger.info(f"Mutated code: \n{code}\n")
    return compile_code_to_plain_function(code), code


# ================= Evolution Loop (async) =================
async def evolve_tool(
    baseline_tool: Callable,
    bench: List[Dict[str, Any]],
    generations: int = 3,
    variants: int = 5,
    max_concurrent_mutations: int = 5,
) -> Tuple[Callable, str]:
    best_tool_like = baseline_tool
    best_score = evaluate_tool(best_tool_like, bench)
    best_code = _clean_src(baseline_tool)

    logger.info(f"Init code: \n{best_code}\n")
    logger.info(f"Init score = {best_score:.2f}\n")

    sem = asyncio.Semaphore(max_concurrent_mutations)

    async def propose_one():
        async with sem:
            return await asyncio.to_thread(mutate_with_claude, best_tool_like)

    for g in range(1, generations + 1):
        logger.info(f"\n=== Generation {g} ===")

        results = await asyncio.gather(
            *[propose_one() for _ in range(variants)], return_exceptions=True
        )

        candidates: List[Tuple[Callable, str, float]] = [
            (baseline_tool, best_code, best_score)
        ]
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Mutation failed: {r}")
                continue
            fn, code = r
            try:
                score = evaluate_tool(fn, bench)
                candidates.append((fn, code, score))
                logger.info(f" → variant score={score:.2f}")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        candidates.sort(key=lambda x: x[2], reverse=True)
        winner_fn, winner_code, winner_score = candidates[0]
        best_tool_like, best_code, best_score = winner_fn, winner_code, winner_score
        logger.info(f"Winner @ Gen {g}: score={best_score:.2f}")

    return get_callable(best_tool_like), best_code


if __name__ == "__main__":
    final_fn, final_code = asyncio.run(
        evolve_tool(search_tool, benchmark, generations=3, variants=5)
    )
    print("\n=== Best evolved tool code ===\n")
    print(final_code)
