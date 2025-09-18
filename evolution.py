#!/usr/bin/env python3
"""
tool_code_evolution.py
Evolve Python @function_tool functions by mutating their code with GPT-4.1 and Claude.
Evaluates each generation on a benchmark, keeps the best variant.
"""

import inspect
import textwrap
import random
import asyncio
import logging

from dotenv import load_dotenv
from openai import OpenAI
# Simple function_tool decorator that just passes through the function

from anthropic import Anthropic
from agents import function_tool


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tool-code-evolution")

# Initialize clients
openai_client = OpenAI()
anthropic_client = Anthropic()

# =============== Baseline Example Tool ===============

@function_tool
def search_tool(query: str, docs: list[str]) -> str:
    """Return the first document containing the query (case-insensitive)."""
    for d in docs:
        if query.lower() in d.lower():
            return d
    return "Not found"

# =============== Benchmark Dataset ===============

benchmark = [
    {"query": "apple", "docs": ["banana", "orange", "apple pie"], "expected": "apple pie"},
    {"query": "machine learning", "docs": ["deep learning basics", "machine learning guide"], "expected": "machine learning guide"},
    {"query": "typo test", "docs": ["typ test doc", "typoooo"], "expected": "typ test doc"},
]


# =============== Evaluation ===============

def evaluate_tool(tool, benchmark):
    correct = 0
    for item in benchmark:
        try:
            out = tool(item["query"], item["docs"])
            if item["expected"].lower() in out.lower():
                correct += 1
        except Exception as e:
            logger.warning(f"Tool crashed on query {item['query']}: {e}")
    return correct / len(benchmark)


# =============== Mutation Functions ===============

def mutate_with_gpt(func):
    src = textwrap.dedent(inspect.getsource(func))
    prompt = f"""
Here is a Python function decorated with @function_tool:

{src}

Modify it into a new variant that better finds the most relevant document,
not just the first match. You may:
- Add typo tolerance (Levenshtein-like logic, fuzzy matching)
- Rank documents by relevance
- Use multiple keywords
- Keep the same function signature and decorator

Return ONLY valid Python code with the @function_tool decorator.
"""

    resp = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )
    code = resp.choices[0].message.content
    return compile_code(code), code


def mutate_with_claude(func):
    src = textwrap.dedent(inspect.getsource(func))
    prompt = f"""
Here is a Python function decorated with @function_tool:

{src}

Please improve this function so that it selects the BEST matching document
for a query (not just the first match). Consider typo tolerance, keyword scoring,
and ranking by overlap.

Return ONLY valid Python code with the @function_tool decorator.
"""

    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    code = resp.content[0].text
    return compile_code(code), code


def compile_code(code_str):
    g = {"function_tool": function_tool}
    l = {}
    exec(code_str, g, l)
    tools = [v for v in l.values() if callable(v)]
    if not tools:
        raise RuntimeError("No function found in mutated code.")
    return tools[0]


# =============== Evolutionary Loop ===============

async def evolve_tool(baseline_tool, benchmark, generations=3, variants=5):
    best_tool = baseline_tool
    best_score = evaluate_tool(best_tool, benchmark)
    best_code = inspect.getsource(baseline_tool)

    for g in range(generations):
        logger.info(f"\n=== Generation {g} | Best score = {best_score:.2f} ===")

        # propose variants asynchronously
        tasks = []
        for _ in range(variants):
            mutator = random.choice([mutate_with_gpt, mutate_with_claude])
            tasks.append(asyncio.to_thread(mutator, best_tool))  # run in thread pool since it's blocking

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = [(best_tool, best_code, best_score)]
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Mutation failed: {r}")
                continue
            f, code = r
            score = evaluate_tool(f, benchmark)
            candidates.append((f, code, score))
            logger.info(f" → Variant score={score:.2f}")

        # pick the best
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_tool, best_code, best_score = candidates[0]
        logger.info(f" → Gen {g} winner = {best_score:.2f}")

    return best_tool, best_code


# =============== Run ===============

if __name__ == "__main__":
    asyncio.run(evolve_tool(search_tool, benchmark, generations=3, variants=5))