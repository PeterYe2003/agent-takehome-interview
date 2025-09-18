#!/usr/bin/env python3
"""
tool_code_evolution.py
Evolve Python @function_tool functions by mutating their code with GPT-4.1 and Claude.
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
from agents import function_tool  # your project's FunctionTool decorator

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tool-code-evolution")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ================= Baseline tool (decorated; used by your app) =================
@function_tool
def search_tool(query: str, docs: list[str]) -> str:
    """Return the first document containing the query (case-insensitive)."""
    for d in docs:
        if query.lower() in d.lower():
            return d
    return "Not found"

# ================= Benchmark =================
benchmark: List[Dict[str, Any]] = [
    {"query": "apple", "docs": ["banana", "orange", "apple pie"], "expected": "apple pie"},
    {"query": "machine learning", "docs": ["deep learning basics", "machine learning guide"], "expected": "machine learning guide"},
    {"query": "typo test", "docs": ["typ test doc", "typoooo"], "expected": "typ test doc"},
]

# ================= Helpers =================
def _strip_md_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:\w+)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
    return code.strip()

def _clean_src_keep_function_tool(func: Callable) -> str:
    """Dedent source and strip non-function_tool decorators."""
    # If it's a FunctionTool, we can't extract source, so return a placeholder
    if hasattr(func, 'on_invoke_tool'):
        return "def search_tool(query: str, docs: list[str]) -> str:\n    # FunctionTool object - source not available\n    pass"
    
    # If it's a regular function, try to get the underlying function's source
    target = getattr(func, "func", func)
    src = textwrap.dedent(inspect.getsource(target))
    lines = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("@") and "function_tool" not in s:
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

def get_callable(tool_obj: Any) -> Callable:
    """
    Return a plain callable for evaluation.
    - If it's a FunctionTool, unwrap underlying python function via .func (or common fallbacks).
    - Else assume it's already a function.
    """
    if hasattr(tool_obj, 'on_invoke_tool'):
        # FunctionTool object - create a wrapper function
        async def wrapper(query: str, docs: list[str]) -> str:
            import json
            params = json.dumps({"query": query, "docs": docs})
            return await tool_obj.on_invoke_tool(None, params)
        
        # Make it synchronous for evaluation
        def sync_wrapper(query: str, docs: list[str]) -> str:
            import asyncio
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, create a new task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, wrapper(query, docs))
                        return future.result()
                else:
                    return asyncio.run(wrapper(query, docs))
            except RuntimeError:
                # No event loop, safe to use asyncio.run
                return asyncio.run(wrapper(query, docs))
        
        return sync_wrapper
    
    for attr in ("func", "_func", "_callable"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    if callable(tool_obj):
        return tool_obj
    raise TypeError("Provided tool is not callable and cannot be unwrapped.")

def compile_code_to_plain_function(code_str: str) -> Callable:
    """
    Compile mutated code as a PLAIN FUNCTION by injecting a passthrough decorator
    named 'function_tool'. This ensures evolved variants are easy to benchmark.
    """
    code = _strip_md_fences(code_str)
    def _passthrough(fn):  # decorator shim
        return fn
    g: Dict[str, Any] = {"function_tool": _passthrough}
    l: Dict[str, Any] = {}
    exec(code, g, l)
    # Return first callable defined
    tools = [v for v in l.values() if callable(v)]
    if not tools:
        raise RuntimeError("No callable found in mutated code.")
    return tools[0]

# ================= Evaluation =================
def evaluate_tool(tool_like: Any, bench: List[Dict[str, Any]]) -> float:
    """
    Evaluate either a FunctionTool or a plain function on the benchmark.
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

# ================= Mutators =================
def mutate_with_gpt(func: Any) -> Tuple[Callable, str]:
    # If it's a FunctionTool, we can't mutate it further
    if hasattr(func, 'on_invoke_tool'):
        raise RuntimeError("could not get source code")
    
    src = _clean_src_keep_function_tool(func)
    prompt = f"""
You are improving a Python tool function.
It MUST keep the SAME NAME and signature and the @function_tool decorator.

Current code:

{src}

Goal: Choose the BEST matching document for a query (not just the first).
You may add:
- light typo tolerance (simple edit distance)
- token/keyword overlap scoring
- ranking & tie-breaking
- deterministic logic (no randomness)
- standard library only (no external deps)

HARD CONSTRAINTS:
- Keep the @function_tool decorator.
- Keep the SAME function name and signature: (query: str, docs: list[str]) -> str
- Return ONLY valid Python code (no backticks, no prose).
"""
    resp = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )
    code = resp.choices[0].message.content or ""
    return compile_code_to_plain_function(code), code

def mutate_with_claude(func: Any) -> Tuple[Callable, str]:
    # If it's a FunctionTool, we can't mutate it further
    if hasattr(func, 'on_invoke_tool'):
        raise RuntimeError("could not get source code")
    
    src = _clean_src_keep_function_tool(func)
    prompt = f"""
Improve the following Python function.
Keep the SAME NAME and signature and the @function_tool decorator.

Code:

{src}

Requirements:
- Rank docs for best match using token overlap and light typo tolerance (edit distance).
- Deterministic, standard library only.
- Return ONLY valid Python code (no markdown fences).
"""
    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    # Claude returns content blocks
    code = "".join(getattr(block, "text", "") for block in (resp.content or []))
    return compile_code_to_plain_function(code), code

# ================= Evolution Loop (async) =================
async def evolve_tool(
    baseline_tool: Any,
    bench: List[Dict[str, Any]],
    generations: int = 3,
    variants: int = 5,
    max_concurrent_mutations: int = 5,
) -> Tuple[Callable, str]:
    best_tool_like = baseline_tool  # may be FunctionTool
    best_score = evaluate_tool(best_tool_like, bench)
    best_code = _clean_src_keep_function_tool(baseline_tool)
    logger.info(f"Init score = {best_score:.2f}")

    sem = asyncio.Semaphore(max_concurrent_mutations)

    async def propose_one(mutator):
        async with sem:
            return await asyncio.to_thread(mutator, best_tool_like)

    for g in range(1, generations + 1):
        logger.info(f"\n=== Generation {g} ===")
        # propose variants concurrently (mix GPT and Claude)
        mutators = [random.choice([mutate_with_gpt, mutate_with_claude]) for _ in range(variants)]
        results = await asyncio.gather(*(propose_one(m) for m in mutators), return_exceptions=True)

        candidates: List[Tuple[Callable, str, float]] = [(get_callable(best_tool_like), best_code, best_score)]
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Mutation failed: {r}")
                continue
            fn, code = r
            try:
                score = evaluate_tool(fn, bench)  # fn is a plain callable
                candidates.append((fn, code, score))
                logger.info(f" → variant score={score:.2f}")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        candidates.sort(key=lambda x: x[2], reverse=True)
        winner_fn, winner_code, winner_score = candidates[0]
        best_tool_like, best_code, best_score = winner_fn, winner_code, winner_score
        logger.info(f"Winner @ Gen {g}: score={best_score:.2f}")

    return get_callable(best_tool_like), best_code

# ================= Run =================
if __name__ == "__main__":
    final_fn, final_code = asyncio.run(
        evolve_tool(search_tool, benchmark, generations=3, variants=5)
    )
    print("\n=== Best evolved tool code ===\n")
    print(final_code)
