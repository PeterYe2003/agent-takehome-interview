#!/usr/bin/env python3
"""
Evolution + Agent System for optimizing search_evidence_for_option_tool
Combines evolution.py and test.py to create an evolutionary system that optimizes
the search function and then creates agents that use the best evolved function.
"""

import os
import time
import logging
import asyncio
import re
import inspect
import textwrap
import random
from dotenv import load_dotenv
from datasets import load_dataset
from agents import Agent, Runner, function_tool, RunContextWrapper
import matplotlib.pyplot as plt
from openai import OpenAI
from anthropic import Anthropic
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable

# Load environment variables (e.g., API keys) from a .env file if present.
load_dotenv()

# Common stop words for keyword extraction
STOP_WORDS = {
    "that", "this", "with", "from", "they", "have", "been", "will", "were", "said",
    "each", "which", "their", "time", "would", "there", "could", "other", "after",
    "first", "well", "also", "new", "want", "because", "any", "these", "give", "day",
    "most", "us", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "shall"
}

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("evolution-agent-system")

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ================= Context and Task Setup =================


@dataclass
class TaskContext:
    """Context object containing all task information."""

    question: str
    context: str
    choices: Dict[str, str]
    answer: str
    task_id: int
    original_context_length: int
    context_category: str

# Categorizes context length into Short, Medium, or Long.
def get_context_length_category(context_length: int) -> str:
    """Categorize context length into Short, Medium, or Long."""
    if context_length <= 144000:
        return "Short (0-144k chars)"
    elif context_length <= 512000:
        return "Medium (144k-512k chars)"
    else:
        return "Long (512k-8M chars)"
# Ensures that the response is only one letter (A, B, C, D).
def _letter_only(s: str) -> str | None:
    """Extract just the letter (A, B, C, D) from the response."""
    m = re.match(r"^\s*([ABCD])\b", s.strip().upper())
    return m.group(1) if m else None



# ================= Benchmark Creation =================


def create_search_benchmark_from_longbench() -> List[Dict[str, Any]]:
    """Create a benchmark for evaluating search_evidence_for_option_tool variants using LongBench-v2."""
    logger.info("Loading LongBench-v2 dataset for search function benchmark...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")

    # Use first 50 entries
    benchmark_tasks = [dataset[i] for i in range(min(50, len(dataset)))]
    logger.info(
        f"Created benchmark with {len(benchmark_tasks)} tasks from LongBench-v2"
    )

    benchmark = []
    for i, task in enumerate(benchmark_tasks):
        # Extract basic info
        context = task.get("context", "") or ""
        question = task["question"]
        choices = {
            "A": task.get("choice_A", ""),
            "B": task.get("choice_B", ""),
            "C": task.get("choice_C", ""),
            "D": task.get("choice_D", ""),
        }
        answer = task["answer"]

        # Create a simple keyword extraction for the correct answer
        correct_option_text = choices.get(answer, "")
        # Simple keyword extraction - take first few meaningful words
        words = re.findall(r"\b\w+\b", correct_option_text.lower())
        meaningful_words = [
            w for w in words if len(w) > 3 and w not in STOP_WORDS
        ]
        keywords = (
            ", ".join(meaningful_words[:3])
            if meaningful_words
            else correct_option_text.lower()[:50]
        )

        benchmark.append(
            {
                "context": context,
                "question": question,
                "choices": choices,
                "answer": answer,
                "option_letter": answer,
                "keywords": keywords,
                "expected_evidence_contains": meaningful_words[:3]
                if meaningful_words
                else [answer.lower()],
            }
        )

    return benchmark


# Original search function for evidence supporting a specific option.
def search_evidence_for_option_tool_original(
    context: str,
    option_letter: str,
    keywords: str,
    choices: Dict[str, str],
    expansion_chars: int = 500,
    max_results: int = 15,
) -> str:
    """Original search function for evidence supporting a specific option."""
    if option_letter not in choices:
        return f"Invalid option letter: {option_letter}"

    option_text = choices[option_letter]

    # Parse keywords from the string (comma-separated)
    keyword_list = [kw.strip().lower() for kw in keywords.split(",") if kw.strip()]

    if not keyword_list:
        return f"No valid keywords provided for searching evidence for option {option_letter}."

    context_lower = context.lower()

    # Find all matches for each keyword with scoring
    scored_matches = []

    for keyword in keyword_list:
        if not keyword or len(keyword) < 2:
            continue

        # Find all occurrences of the keyword
        start = 0
        while True:
            pos = context_lower.find(keyword, start)
            if pos == -1:
                break

            # Calculate expansion boundaries
            expand_start = max(0, pos - expansion_chars // 2)
            expand_end = min(len(context), pos + len(keyword) + expansion_chars // 2)

            # Extract the expanded text
            expanded_text = context[expand_start:expand_end]

            # Add context markers
            if expand_start > 0:
                expanded_text = "..." + expanded_text
            if expand_end < len(context):
                expanded_text = expanded_text + "..."

            # Calculate relevance score for this match
            text_lower = expanded_text.lower()
            score = 0

            # Base score for finding the keyword
            score += 1

            # Bonus for multiple keyword matches in the same section
            for other_keyword in keyword_list:
                if other_keyword != keyword and other_keyword in text_lower:
                    score += 3

            # Bonus for option-specific terms appearing in context
            option_words = re.findall(r"\b\w+\b", option_text.lower())
            for ow in option_words:
                if len(ow) > 3 and ow in text_lower:
                    score += 2

            scored_matches.append(
                {
                    "keyword": keyword,
                    "position": pos,
                    "text": expanded_text.strip(),
                    "score": score,
                    "keywords_found": [kw for kw in keyword_list if kw in text_lower],
                    "option_terms_found": [
                        ow for ow in option_words if len(ow) > 3 and ow in text_lower
                    ],
                }
            )

            start = pos + 1

    if not scored_matches:
        return f"No evidence found for option {option_letter} with keywords: {', '.join(keyword_list)}"

    # Sort by score (highest first), then by position
    scored_matches.sort(key=lambda x: (-x["score"], x["position"]))

    # Remove duplicates and select top results
    unique_texts = []
    seen_texts = set()

    for match in scored_matches:
        # Use a normalized version for deduplication
        normalized_text = re.sub(r"\s+", " ", match["text"].lower().strip())

        if normalized_text not in seen_texts and len(unique_texts) < max_results:
            keywords_found = ", ".join(match["keywords_found"])
            option_terms = ", ".join(match["option_terms_found"])
            unique_texts.append(
                f"[Score: {match['score']:.1f}, Keywords: {keywords_found}, Option terms: {option_terms}]: {match['text']}"
            )
            seen_texts.add(normalized_text)

    if not unique_texts:
        return f"No unique evidence found for option {option_letter} with keywords: {', '.join(keyword_list)}"

    result = (
        f"Evidence for Option {option_letter} ({len(unique_texts)} sections found):\n\n"
        + "\n\n".join(unique_texts)
    )
    return result


# ================= Evolution System =================

# Helper function to remove markdown fences from generated code.
def _strip_md_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:\w+)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
    return code.strip()

# converts function to string and removes decorators.
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
    """Return a plain callable for evaluation."""
    for attr in ("func", "_func", "_callable"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    if callable(tool_obj):
        return tool_obj
    raise TypeError("Provided tool is not callable and cannot be unwrapped.")


def compile_code_to_plain_function(code_str: str) -> Callable:
    """turns claude-mutated code into a function"""
    code = _strip_md_fences(code_str)
    g: Dict[str, Any] = {}
    l: Dict[str, Any] = {}

    exec(code, g, l)

    tools = [v for v in l.values() if callable(v)]
    if not tools:
        raise RuntimeError("No callable found in mutated code.")
    return tools[0]


def create_search_tool_function(search_func: Callable):
    """Create a function tool that wraps the search function and accesses context from TaskContext."""
    
    @function_tool
    def search_evidence_for_option_tool(
        wrapper: RunContextWrapper[TaskContext],
        option_letter: str,
        keywords: str,
        expansion_chars: int = 500,
        max_results: int = 15,
    ) -> str:
        """Search for evidence supporting a specific option using the evolved search function."""
        # Get context from the wrapper
        task_context = wrapper.context
        context = task_context.context
        choices = task_context.choices
        
        # Call the evolved search function
        return search_func(
            context=context,
            option_letter=option_letter,
            keywords=keywords,
            choices=choices,
            expansion_chars=expansion_chars,
            max_results=max_results,
        )
    
    return search_evidence_for_option_tool


def create_other_tools():
    """Create other necessary tools for the agent."""
    # Return the standalone tool functions that are defined later in the file
    return [analyze_question_type_tool, get_full_context_tool, extract_question_keywords_tool]


async def evaluate_search_function(
    search_func: Callable, benchmark: List[Dict[str, Any]]
) -> float:
    """Evaluate a search function by creating an agent with it and testing accuracy."""
    # Create an agent with the search function
    search_tool = create_search_tool_function(search_func)
    other_tools = create_other_tools()
    
    agent = Agent[TaskContext](
        name="search-evaluation-agent",
        instructions="""You are a systematic question-answering agent for multiple-choice questions that uses evidence-based analysis.

Your systematic process:
1. FIRST: Use analyze_question_type_tool to understand the question type, complexity, and context characteristics
2. SECOND: If context is SHORT (≤100k chars), use get_full_context_tool to read the entire context directly
3. THIRD: If context is MEDIUM/LONG, use extract_question_keywords_tool to identify keywords from the question
4. FOURTH: For MEDIUM/LONG contexts, use search_evidence_for_option_tool for each option (A, B, C, D) with appropriate parameters
5. FIFTH: Compare the evidence found for each option and select the one with the strongest supporting evidence
6. SIXTH: Respond with ONLY the letter of your chosen answer (A, B, C, or D)

Remember: Your response must be exactly one letter (A, B, C, or D).""",
        model="gpt-4o-mini",  # Use faster model for evaluation
        tools=[search_tool] + other_tools
    )
    
    runner = Runner()
    correct = 0
    total = 0
    
    # Evaluate on a subset of benchmark for speed (first 10 items)
    eval_items = benchmark[:10]
    
    for i, item in enumerate(eval_items):
        try:
            # Create task context
            context = item["context"]
            question = item["question"]
            choices = item["choices"]
            answer = item["answer"]
            
            original_context_length = len(context)
            context_category = get_context_length_category(original_context_length)

            task_context = TaskContext(
                question=question,
                context=context,
                choices=choices,
                answer=answer,
                task_id=i,
                original_context_length=original_context_length,
                context_category=context_category
            )

            # Create input prompt (without context - it's stored in TaskContext)
            input_prompt = (
                f"Question:\n{question}\n\n"
                "Choices:\n"
                f"A) {choices['A']}\n"
                f"B) {choices['B']}\n"
                f"C) {choices['C']}\n"
                f"D) {choices['D']}\n\n"
                "Use the available tools to analyze the context and find evidence for your answer. Follow your systematic process.\n"
                "Answer:"
            )

            # Run the agent
            result = await runner.run(agent, input=input_prompt, context=task_context)
            raw_output = (result.final_output or "").strip()
            letter = _letter_only(raw_output)
            is_correct = (letter == answer)
            
            if is_correct:
                correct += 1
            total += 1
            
            logger.debug(f"Search function evaluation - Item {i}: {'✓' if is_correct else '✗'} (expected: {answer}, got: {letter})")
            
        except Exception as e:
            logger.warning(f"Search function evaluation failed on item {i}: {e}")
            total += 1  # Count as incorrect

    # Return accuracy (0-1)
    accuracy = (correct / total) if total > 0 else 0.0
    logger.info(f"Search function evaluation: {correct}/{total} = {accuracy:.3f}")
    return accuracy


def mutate_search_function_with_claude(func: Any) -> Tuple[Callable, str]:
    src = _clean_src(func)
    prompt = f"""
Improve the following Python search function for finding evidence in text.
Keep the SAME NAME and signature.

Code:

{src}

Requirements:
- Improve the search algorithm to find more relevant evidence
- Better scoring mechanism for relevance
- More robust keyword matching
- Better handling of context expansion
- Deterministic, standard library only
- Return ONLY valid Python code (no markdown fences)
- DO NOT use any excess text like "Here's an improved version..."
- Have ONLY the code in your response as if it was in a .py file
"""
    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    code = "".join(getattr(block, "text", "") for block in (resp.content or []))
    logger.info(f"Mutated search function code: \n{code}\n")
    return compile_code_to_plain_function(code), code


async def evolve_search_function(
    baseline_search_func: Callable,
    benchmark: List[Dict[str, Any]],
    generations: int = 3,
    variants: int = 5,
    max_concurrent_mutations: int = 5,
) -> Tuple[Callable, str]:
    """Evolve the search function using the benchmark."""
    best_search_func = baseline_search_func
    best_score = await evaluate_search_function(best_search_func, benchmark)
    best_code = _clean_src(baseline_search_func)

    logger.info(f"Initial search function score = {best_score:.3f}")
    logger.info(f"Initial search function code:\n{best_code}")

    sem = asyncio.Semaphore(max_concurrent_mutations)

    async def propose_one():
        async with sem:
            return await asyncio.to_thread(
                mutate_search_function_with_claude, best_search_func
            )

    for g in range(1, generations + 1):
        logger.info(f"\n=== Search Function Evolution Generation {g} ===")

        results = await asyncio.gather(
            *[propose_one() for _ in range(variants)], return_exceptions=True
        )

        candidates: List[Tuple[Callable, str, float]] = [
            (baseline_search_func, best_code, best_score)
        ]
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Search function mutation failed: {r}")
                continue
            fn, code = r
            try:
                score = await evaluate_search_function(fn, benchmark)
                candidates.append((fn, code, score))
                logger.info(f" → search variant score={score:.3f}")
                logger.info(f" → search variant code:\n{code}")
            except Exception as e:
                logger.warning(f"Search function evaluation failed: {e}")

        candidates.sort(key=lambda x: x[2], reverse=True)
        winner_fn, winner_code, winner_score = candidates[0]
        best_search_func, best_code, best_score = winner_fn, winner_code, winner_score
        logger.info(f"Best search function @ Gen {g}: score={best_score:.3f}")
        logger.info(f"Best search function code @ Gen {g}:\n{best_code}")

    return get_callable(best_search_func), best_code


# ================= Agent Creation =================

@function_tool
def extract_question_keywords_tool(wrapper: RunContextWrapper[TaskContext]) -> str:
    """Extract the most important keywords from the question using LLM analysis."""
    task_context = wrapper.context

    # Use the LLM to identify the most important keywords in the question
    prompt = f"""Analyze this question and identify the 3-5 most important keywords that would be crucial for finding relevant information in a long context.

Question: {task_context.question}

Focus on:
- Key concepts, entities, or topics
- Important verbs or actions
- Specific terms that would appear in supporting evidence
- Avoid common words like "what", "which", "the", "is", etc.

Return only the keywords separated by commas, no explanations."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        keywords = response.choices[0].message.content.strip()

        # Save to combined keywords file for debugging
        with open(f"keywords_task_{task_context.task_id}.txt", "w") as f:
            f.write(f"Task ID: {task_context.task_id}\n")
            f.write(f"Question: {task_context.question}\n")
            f.write(f"Question Keywords: {keywords}\n")
            f.write(f"Choices: {task_context.choices}\n")
            f.write(f"Answer: {task_context.answer}\n")
            f.write(f"Context Length: {len(task_context.context)} chars\n")
            f.write(f"Context Category: {task_context.context_category}\n\n")

        return f"Question keywords: {keywords}"
    except Exception as e:
        logger.warning(
            f"Failed to extract question keywords for task {task_context.task_id}: {e}"
        )
        return "Failed to extract question keywords"

@function_tool
def extract_option_keywords_tool(
    wrapper: RunContextWrapper[TaskContext], option_letter: str
) -> str:
    """Extract the most important keywords from a specific option using LLM analysis."""
    task_context = wrapper.context

    if option_letter not in task_context.choices:
        return f"Invalid option letter: {option_letter}"

    option_text = task_context.choices[option_letter]

    # Use the LLM to identify the most important keywords in this specific option
    prompt = f"""Analyze this answer choice and identify the 2-4 most important keywords that would be crucial for finding supporting evidence in a long context.

Answer Choice {option_letter}: {option_text}

Focus on:
- Key concepts, entities, or specific terms unique to this option
- Important details that would distinguish this option from others
- Terms that would appear in evidence supporting this choice
- Avoid common words like "the", "is", "and", etc.

Return only the keywords separated by commas, no explanations."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        keywords = response.choices[0].message.content.strip()

        # Append to combined keywords file for debugging
        with open(f"keywords_task_{task_context.task_id}.txt", "a") as f:
            f.write(f"Option {option_letter} Keywords: {keywords}\n")

        return f"Option {option_letter} keywords: {keywords}"
    except Exception as e:
        logger.warning(
            f"Failed to extract option {option_letter} keywords for task {task_context.task_id}: {e}"
        )
        return f"Failed to extract option {option_letter} keywords"

@function_tool
def analyze_question_type_tool(wrapper: RunContextWrapper[TaskContext]) -> str:
    """Analyze the question type and context to help the LLM decide search parameters."""
    task_context = wrapper.context
    context_length = len(task_context.context)
    question = task_context.question

    # Categorize context length
    if context_length <= 100000:
        context_category = "Short"
    elif context_length <= 500000:
        context_category = "Medium"
    elif context_length <= 1000000:
        context_category = "Long"
    else:
        context_category = "Very Long"

    # Analyze question type using LLM
    analysis_prompt = f"""Analyze this question and determine its type and complexity:

Question: {question}

Classify this question as one of the following types:
1. FACTUAL: Simple fact retrieval, definitions, specific details
2. ANALYTICAL: Requires reasoning, comparison, analysis, interpretation
3. COMPREHENSIVE: Requires understanding multiple concepts, relationships, or complex processes
4. INFERENTIAL: Requires drawing conclusions from evidence or making connections

Also assess:
- Complexity level (1-5 scale, where 5 is most complex)
- Whether it requires broad context understanding or narrow focus
- Whether multiple pieces of information need to be connected

Respond in this format:
Type: [FACTUAL/ANALYTICAL/COMPREHENSIVE/INFERENTIAL]
Complexity: [1-5]
Context_Need: [BROAD/NARROW]
Reasoning: [Brief explanation]"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": analysis_prompt}],
        max_tokens=150,
        temperature=0.1,
    )
    analysis = response.choices[0].message.content.strip()

    # Save analysis to file for debugging
    with open(f"question_analysis_task_{task_context.task_id}.txt", "w") as f:
        f.write(f"Task ID: {task_context.task_id}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Context Length: {context_length:,} characters\n")
        f.write(f"Context Category: {context_category}\n\n")
        f.write(f"Question Analysis:\n{analysis}\n")

        info = f"""Context and Question Analysis:
- Context Length: {context_length:,} characters
- Context Category: {context_category}

Question Analysis:
{analysis}

Parameter Selection Guidelines (MAXIMIZE CONTEXT USAGE):
- FACTUAL questions: Use expansion_chars (800-2000) and max_results (20-80)
- ANALYTICAL questions: Use expansion_chars (1200-3000) and max_results (40-120)
- COMPREHENSIVE questions: Use expansion_chars (2000-4000) and max_results (60-150)
- INFERENTIAL questions: Use expansion_chars (2500-5000) and max_results (80-200)

Context Usage Strategy:
- SHORT contexts (≤100k chars): Use 60-80% of available context
* expansion_chars: 800-2000, max_results: 20-80
- MEDIUM contexts (100k-500k chars): Use 70-90% of available context
* expansion_chars: 1200-3000, max_results: 40-120
- LONG contexts (500k-1M chars): Use 80-95% of available context
* expansion_chars: 2000-4000, max_results: 60-150
- VERY LONG contexts (>1M chars): Use 90-98% of available context
* expansion_chars: 2500-5000, max_results: 80-200

Token Limit Considerations:
- 128k token limit ≈ 512k characters
- Each result with expansion ≈ 1000-5000 characters
- Be aggressive with parameters but stay under limit
- Prioritize max_results over expansion_chars for comprehensive coverage

Choose parameters that MAXIMIZE context usage while staying within token limits."""

        return info

@function_tool
def get_full_context_tool(wrapper: RunContextWrapper[TaskContext]) -> str:
    """Get the full context for short contexts that can be read entirely."""
    task_context = wrapper.context
    context = task_context.context

    # Save full context to file for debugging
    try:
        with open(f"full_context_task_{task_context.task_id}.txt", "w") as f:
            f.write(f"Task ID: {task_context.task_id}\n")
            f.write(f"Question: {task_context.question}\n")
            f.write(f"Choices: {task_context.choices}\n")
            f.write(f"Answer: {task_context.answer}\n")
            f.write(f"Context Length: {len(context)} chars\n")
            f.write(f"Context Category: {task_context.context_category}\n\n")
            f.write("Full Context:\n")
            f.write("=" * 60 + "\n")
            f.write(context)
    except Exception as e:
        logger.warning(
            f"Failed to save full context for task {task_context.task_id}: {e}"
        )

    return f"Full context ({len(context):,} characters):\n\n{context}"


def create_agent_with_evolved_search(
    evolved_search_func: Callable,
) -> Agent[TaskContext]:
    """Create an agent that uses the evolved search function."""
    search_tool = create_search_tool_function(evolved_search_func)

    agent = Agent[TaskContext](
        name="evolved-systematic-evidence-agent",
        instructions="""You are a systematic question-answering agent for multiple-choice questions that uses evidence-based analysis.

Your systematic process:
1. FIRST: Use analyze_question_type_tool to understand the question type, complexity, and context characteristics
2. SECOND: If context is SHORT (≤100k chars), use get_full_context_tool to read the entire context directly
3. THIRD: If context is MEDIUM/LONG, use extract_question_keywords_tool to identify keywords from the question
4. FOURTH: For MEDIUM/LONG contexts, for each option (A, B, C, D), use extract_option_keywords_tool to get keywords specific to that option
5. FIFTH: For MEDIUM/LONG contexts, for each option, use search_evidence_for_option_tool with the question keywords + option keywords and CHOOSE parameters based on question type analysis
6. SIXTH: Compare the evidence found for each option and select the one with the strongest supporting evidence
7. SEVENTH: Respond with ONLY the letter of your chosen answer (A, B, C, or D)

Tool Usage:
- analyze_question_type_tool: Analyzes question type, complexity, and provides parameter guidance
- get_full_context_tool: Gets the complete context for short contexts (≤100k chars)
- extract_question_keywords_tool: Gets 3-5 most important keywords from the question (for medium/long contexts)
- extract_option_keywords_tool(option_letter): Gets 2-4 keywords specific to option A, B, C, or D (for medium/long contexts)
- search_evidence_for_option_tool(option_letter, keywords, expansion_chars, max_results): Searches for evidence with your chosen parameters (for medium/long contexts)

Systematic Workflow:
1. Call analyze_question_type_tool to understand question type, complexity, and context characteristics
2. If context is SHORT (≤100k chars):
   a. Call get_full_context_tool to read the entire context
   b. Analyze the full context directly to answer the question
3. If context is MEDIUM/LONG (>100k chars):
   a. Call extract_question_keywords_tool to get question keywords
   b. For each option (A, B, C, D):
      i. Call extract_option_keywords_tool with the option letter
      ii. Combine question keywords + option keywords
      iii. Call search_evidence_for_option_tool with the combined keywords AND choose parameters based on:
          - Question type (FACTUAL/ANALYTICAL/COMPREHENSIVE/INFERENTIAL)
          - Complexity level (1-5)
          - Context need (BROAD/NARROW)
          - Context length and characteristics
4. Compare the evidence quality and quantity for each option
5. Select the option with the strongest, most relevant evidence
6. Respond with only the letter (A, B, C, or D)

Remember: Your response must be exactly one letter (A, B, C, or D).""",
        model="gpt-4.1",
        tools=[search_tool] + [extract_question_keywords_tool,
        extract_option_keywords_tool,
        analyze_question_type_tool,
        get_full_context_tool]
    )
    return agent


# ================= Agent Evaluation =================


def prepare_longbench2(n=None):
    """Prepare a sample from LongBench-v2 dataset."""
    if n is None:
        n = int(os.getenv("N", "10"))  # Default to 10 for faster testing
    logger.info("Preparing dataset sample: n=%d", n)
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    total = len(dataset)
    if n > total:
        n = total
    indices = random.sample(range(total), n)
    sample = [dataset[i] for i in indices]
    logger.info("Prepared %d tasks from LongBench-v2", len(sample))
    return sample


async def evaluate_agent(
    agent: Agent[TaskContext], tasks: List[dict], max_tasks: int = 5
) -> float:
    """Evaluate an agent on a subset of tasks and return accuracy."""
    runner = Runner()

    # Use only a subset for faster evaluation
    eval_tasks = tasks[:max_tasks]
    correct = 0
    total = 0

    for i, task in enumerate(eval_tasks):
        try:
            q = task["question"]
            context = task.get("context", "") or ""
            choices = {
                "A": task.get("choice_A", ""),
                "B": task.get("choice_B", ""),
                "C": task.get("choice_C", ""),
                "D": task.get("choice_D", ""),
            }

            original_context_length = len(context)
            context_category = get_context_length_category(original_context_length)

            task_context = TaskContext(
                question=q,
                context=context,
                choices=choices,
                answer=task["answer"],
                task_id=i,
                original_context_length=original_context_length,
                context_category=context_category,
            )

            input_prompt = (
                f"Question:\n{q}\n\n"
                "Choices:\n"
                f"A) {choices['A']}\n"
                f"B) {choices['B']}\n"
                f"C) {choices['C']}\n"
                f"D) {choices['D']}\n\n"
                "Answer:"
            )

            result = await runner.run(agent, input=input_prompt, context=task_context)
            raw_output = (result.final_output or "").strip()
            letter = _letter_only(raw_output)
            is_correct = letter == task["answer"]

            if is_correct:
                correct += 1
            total += 1

            logger.info(
                f"Task {i}: {'✓' if is_correct else '✗'} (expected: {task['answer']}, got: {letter})"
            )

        except Exception as e:
            logger.warning(f"Task {i} failed: {e}")
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Agent evaluation: {correct}/{total} = {accuracy:.3f}")
    return accuracy


# ================= Main Evolution Loop =================


async def main_evolution_loop(
    search_generations: int = 3,
    search_variants: int = 3,
    agent_generations: int = 3,
    max_eval_tasks: int = 10,
):
    """Main evolution loop that evolves search functions and creates agents."""
    logger.info("Starting main evolution loop")

    # Create benchmark for search function evolution
    search_benchmark = create_search_benchmark_from_longbench()
    logger.info(f"Created search benchmark with {len(search_benchmark)} test cases")

    # Prepare evaluation tasks
    eval_tasks = prepare_longbench2(max_eval_tasks * 2)  # Get more tasks than needed
    logger.info(f"Prepared {len(eval_tasks)} evaluation tasks")

    best_agent = None
    best_accuracy = 0.0
    best_search_code = ""

    for agent_gen in range(1, agent_generations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"AGENT GENERATION {agent_gen}")
        logger.info(f"{'='*60}")

        # Evolve the search function
        logger.info("Evolving search function...")
        evolved_search_func, evolved_search_code = await evolve_search_function(
            baseline_search_func=search_evidence_for_option_tool_original,
            benchmark=search_benchmark,
            generations=search_generations,
            variants=search_variants,
        )

        # Create agent with evolved search function
        logger.info("Creating agent with evolved search function...")
        agent = create_agent_with_evolved_search(evolved_search_func)

        # Evaluate the agent
        logger.info("Evaluating agent...")
        accuracy = await evaluate_agent(agent, eval_tasks, max_eval_tasks)

        # Keep track of the best agent
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_agent = agent
            best_search_code = evolved_search_code
            logger.info(f"NEW BEST AGENT! Accuracy: {accuracy:.3f}")
        else:
            logger.info(
                f"Agent accuracy: {accuracy:.3f} (best so far: {best_accuracy:.3f})"
            )

    logger.info(f"\n{'='*60}")
    logger.info("EVOLUTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best agent accuracy: {best_accuracy:.3f}")
    logger.info(f"Best search function code:")
    logger.info(f"{best_search_code}")

    return best_agent, best_accuracy, best_search_code


if __name__ == "__main__":
    asyncio.run(main_evolution_loop())