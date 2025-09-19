import os
import time
import logging
import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
import random
from agents import Agent, Runner, function_tool, RunContextWrapper
import matplotlib.pyplot as plt
import json
import re
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict


load_dotenv()


logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("agents-test")

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Helper function to categorize context length.
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


client = OpenAI()


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


# Analyzes the question type and context to help the LLM decide search parameters.
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
Context_Need: [BROAD/NARROW]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=150,
            temperature=0,
        )
        analysis = response.choices[0].message.content.strip()

        # Save analysis to file for debugging
        if task_context.task_id % 10 == 0:
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
  * expansion_chars: 400-1000, max_results: 30-100
- MEDIUM contexts (100k-500k chars): Use 70-90% of available context
  * expansion_chars: 600-1500, max_results: 50-150
- LONG contexts (500k-1M chars): Use 80-95% of available context
  * expansion_chars: 1000-2000, max_results: 80-200
- VERY LONG contexts (>1M chars): Use 90-98% of available context
  * expansion_chars: 1250-2500, max_results: 80-150

Token Limit Considerations:
- 128k token limit ≈ 512k characters
- Each result with expansion ≈ 1000-5000 characters
- Be aggressive with parameters but stay under limit
- Prioritize max_results over expansion_chars for comprehensive coverage

Choose parameters that MAXIMIZE context usage while staying within token limits."""

        return info
    except Exception as e:
        logger.warning(
            f"Failed to analyze question type for task {task_context.task_id}: {e}"
        )
        return f"Context Length: {context_length:,} characters, Category: {context_category}. Use standard parameters."

# Extracts keywords from question and all options/answers, then finds relevant text sections.
@function_tool
def extract_keywords_and_find_relevant_text_tool(
    wrapper: RunContextWrapper[TaskContext],
    expansion_chars: int = 1000,
    max_results: int = 20,
) -> str:
    """Extract keywords from question and all options, then find relevant text sections."""
    task_context = wrapper.context

    # Combine question and all options for comprehensive keyword extraction
    question_and_options = f"Question: {task_context.question}\n\n"
    for letter, choice in task_context.choices.items():
        question_and_options += f"Option {letter}: {choice}\n"

    # Use the LLM to identify the most important keywords from the entire question and all options
    prompt = f"""Analyze this multiple-choice question and all answer options to identify the 8-12 most important keywords that would be crucial for finding relevant information in a long context. Try to have at least one unique keyword from each answer option.

{question_and_options}

Focus on:
- Key concepts, entities, or topics from the question
- Important terms from each answer option that would distinguish them
- Specific terms that would appear in supporting evidence
- Avoid common words like "what", "which", "the", "is", "and", etc.
- Prioritize terms that are unique or specific to this question

Return only the keywords separated by commas, no explanations."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        keywords = response.choices[0].message.content.strip()

        # Parse keywords from the string (comma-separated)
        keyword_list = [kw.strip().lower() for kw in keywords.split(",") if kw.strip()]

        if not keyword_list:
            return "No valid keywords extracted."

        # Search for relevant text sections using the keywords
        context = task_context.context
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
                        score += 2  # Bonus for multiple keywords

                # Bonus for question-relevant terms appearing in context
                question_words = re.findall(r"\b\w+\b", task_context.question.lower())
                for qw in question_words:
                    if len(qw) > 3 and qw in text_lower:
                        score += 1

                scored_matches.append(
                    {
                        "keyword": keyword,
                        "position": pos,
                        "text": expanded_text.strip(),
                        "score": score,
                        "keywords_found": [kw for kw in keyword_list if kw in text_lower],
                    }
                )

                start = pos + expansion_chars//2

        if not scored_matches:
            return f"No relevant text found with keywords: {', '.join(keyword_list)}"

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
                unique_texts.append(
                    f"[Score: {match['score']:.1f}, Keywords: {keywords_found}]: {match['text']}"
                )
                seen_texts.add(normalized_text)

        if not unique_texts:
            return f"No unique relevant text found with keywords: {', '.join(keyword_list)}"

        # Save results to file for debugging
        if task_context.task_id % 10 == 0:
            with open(f"relevant_text_task_{task_context.task_id}.txt", "w") as f:
                f.write(f"Task ID: {task_context.task_id}\n")
                f.write(f"Question: {task_context.question}\n")
                f.write(f"Choices: {task_context.choices}\n")
                f.write(f"Answer: {task_context.answer}\n")
                f.write(f"Context Length: {len(context)} chars\n")
                f.write(f"Context Category: {task_context.context_category}\n\n")
                f.write(f"Extracted Keywords: {', '.join(keyword_list)}\n")
                f.write(f"Expansion Chars: {expansion_chars}\n")
                f.write(f"Max Results: {max_results}\n")
                f.write(f"Total Matches Found: {len(scored_matches)}\n")
                f.write(f"Unique Results Returned: {len(unique_texts)}\n\n")
                f.write("Relevant Text Sections (by relevance score):\n")
                f.write("=" * 60 + "\n")
                for i, text in enumerate(unique_texts, 1):
                    f.write(f"\nSection {i}:\n{text}\n")
                    f.write("-" * 40 + "\n")
        result = (
            f"Relevant text sections found ({len(unique_texts)} sections):\n\n"
            + "\n\n".join(unique_texts)
        )
        return result

    except Exception as e:
        logger.warning(
            f"Failed to extract keywords and find relevant text for task {task_context.task_id}: {e}"
        )
        return f"Error in keyword extraction and text search: {e}"


# Gets the full context for short contexts that can be read entirely.
@function_tool
def get_full_context_tool(wrapper: RunContextWrapper[TaskContext]) -> str:
    """Get the full context for short contexts that can be read entirely."""
    task_context = wrapper.context
    context = task_context.context

    # Save full context to file for debugging
    if task_context.task_id % 10 == 0:
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

    return f"Full context ({len(context):,} characters):\n\n{context}"



agent = Agent[TaskContext](
    name="systematic-evidence-agent",
    instructions="""You are a systematic question-answering agent for multiple-choice questions that uses evidence-based analysis.

Your systematic process:
1. FIRST: Use analyze_question_type_tool to understand the question type, complexity, and context characteristics
2. SECOND: If context is SHORT (≤100k chars), use get_full_context_tool to read the entire context directly
3. THIRD: If context is MEDIUM/LONG, use extract_keywords_and_find_relevant_text_tool to identify keywords from question and all options, then find relevant text sections
4. FOURTH: Analyze the relevant text sections to determine which option is best supported by the evidence
5. FIFTH: Respond with ONLY the letter of your chosen answer (A, B, C, or D)

Tool Usage:
- analyze_question_type_tool: Analyzes question type, complexity, and provides parameter guidance
- get_full_context_tool: Gets the complete context for short contexts (≤100k chars)
- extract_keywords_and_find_relevant_text_tool(expansion_chars, max_results): Extracts keywords from question and all options, then finds relevant text sections (for medium/long contexts)

Systematic Workflow:
1. Call analyze_question_type_tool to understand question type, complexity, and context characteristics
2. If context is SHORT (≤100k chars):
   a. Call get_full_context_tool to read the entire context
   b. Analyze the full context directly to answer the question
3. If context is MEDIUM/LONG (>100k chars):
   a. Call extract_keywords_and_find_relevant_text_tool with parameters based on:
      - Question type (FACTUAL/ANALYTICAL/COMPREHENSIVE/INFERENTIAL)
      - Complexity level (1-5)
      - Context need (BROAD/NARROW)
      - Context length and characteristics
   b. This tool will extract keywords from both question and all options, then find relevant text sections
4. Analyze the relevant text sections to determine which option is best supported by the evidence
5. Respond with only the letter (A, B, C, or D)

Question-Type-Based Parameter Selection (MAXIMIZE CONTEXT USAGE):
- FACTUAL questions: Aggressive search (expansion_chars: 800-2000, max_results: 20-80)
  * Use substantial context for comprehensive fact checking
  * Maximize results to ensure no relevant facts are missed
- ANALYTICAL questions: Extensive search (expansion_chars: 1200-3000, max_results: 40-120)
  * Need maximum context for thorough reasoning and comparison
  * Use large expansion and many results for complete analysis
- COMPREHENSIVE questions: Maximum search (expansion_chars: 2000-4000, max_results: 60-150)
  * Require understanding of all concepts and relationships
  * Use largest expansion and most results for complete coverage
- INFERENTIAL questions: Ultra-extensive search (expansion_chars: 2500-5000, max_results: 80-200)
  * Need maximum possible context for drawing conclusions
  * Use maximum expansion and results for comprehensive evidence gathering

Context Maximization Strategy:
- Stay within 128k token limit (≈512k characters)
- Be aggressive with parameters to capture all relevant information

Evidence Analysis:
- Look for sections with high relevance scores
- Prefer evidence that contains multiple keywords from both question and option
- Consider the specificity and directness of the evidence
- Higher scores indicate more relevant evidence
- Use more results and larger expansion for longer contexts to ensure comprehensive coverage

Example:
Question: What is the capital of France?
A) London
B) Paris
C) Berlin
D) Madrid

Process:
1. Get context info and recommended parameters
2. Question keywords: "capital, France"
3. Option A keywords: "London, England"
4. Option B keywords: "Paris, French"
5. Search evidence for each option with appropriate parameters
6. Select option with strongest evidence

Remember: Your response must be exactly one letter (A, B, C, or D).""",
    model="gpt-4.1",
    tools=[
        analyze_question_type_tool,
        get_full_context_tool,
        extract_keywords_and_find_relevant_text_tool,
    ],
)

runner = Runner()

# Randomly samples a subset of the LongBench-v2 dataset.
def prepare_longbench2(n=None):
    if n is None:
        n = int(os.getenv("N", "100"))  # Default to 100 for testing
    logger.info("Preparing dataset sample: n=%d", n)
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    total = len(dataset)
    if n > total:
        n = total
    indices = random.sample(range(total), n)
    sample = [dataset[i] for i in indices]
    logger.info("Prepared %d tasks from LongBench-v2", len(sample))
    return sample


async def run_task(i: int, task: dict):
    question = task["question"]
    context = task.get("context", "") or ""
    choices = {
        "A": task.get("choice_A", ""),
        "B": task.get("choice_B", ""),
        "C": task.get("choice_C", ""),
        "D": task.get("choice_D", ""),
    }

    original_context_length = len(context)
    context_category = get_context_length_category(original_context_length)

    # Create the task context object with the full context
    # The search tools will handle finding relevant sections
    task_context = TaskContext(
        question=question,
        context=context,  # Use full context - the search tools handle filtering
        choices=choices,
        answer=task["answer"],
        task_id=i,
        original_context_length=original_context_length,
        context_category=context_category,
    )

    # Build a simple prompt that doesn't include the context (it's now in local context)
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

    start_time = time.perf_counter()
    logger.debug(
        "Task %s: starting; prompt_len=%d chars (~%d tokens), original_ctx=%d chars",
        i,
        len(input_prompt),
        len(input_prompt) // 4,
        original_context_length,
    )

    result = await runner.run(agent, input=input_prompt, context=task_context)
    raw_output = (result.final_output or "").strip()
    letter = _letter_only(raw_output)
    is_correct = letter == task["answer"]
    duration = time.perf_counter() - start_time
    logger.info("Task %s completed in %.2fs; correct=%s", i, duration, is_correct)
    return {
        "id": i,
        "question": question,
        "choices": choices,
        "expected": task["answer"],
        "output": letter,
        "raw_output": raw_output,
        "correct": is_correct,
        "context_length": original_context_length,
        "context_category": context_category,
        "duration": duration,
    }

# Early visualization of accuracy by context length category
def create_accuracy_chart(category_accuracy, category_stats):
    """Create a bar chart showing accuracy by context length category."""
    categories = list(category_accuracy.keys())
    accuracies = [category_accuracy[cat] * 100 for cat in categories]
    task_counts = [category_stats[cat]["total"] for cat in categories]

    # Create figure with subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy chart
    bars1 = ax1.bar(categories, accuracies, color=["#2E8B57", "#FF8C00", "#DC143C"])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by Context Length Category")
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )

    # Task count chart
    bars2 = ax2.bar(categories, task_counts, color=["#2E8B57", "#FF8C00", "#DC143C"])
    ax2.set_ylabel("Number of Tasks")
    ax2.set_title("Task Distribution by Context Length Category")

    # Add value labels on bars
    for bar, count in zip(bars2, task_counts):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{count}",
            ha="center",
            va="bottom",
        )

    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("accuracy_by_context_length.png", dpi=300, bbox_inches="tight")
    logger.info("Chart saved as 'accuracy_by_context_length.png'")

    # Also create a combined chart
    _, ax = plt.subplots(figsize=(12, 8))

    # Create bars with accuracy and task count
    x = range(len(categories))
    width = 0.35

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        accuracies,
        width,
        label="Accuracy (%)",
        color="#2E8B57",
    )
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        [i + width / 2 for i in x],
        task_counts,
        width,
        label="Task Count",
        color="#FF8C00",
    )

    ax.set_xlabel("Context Length Category")
    ax.set_ylabel("Accuracy (%)", color="#2E8B57")
    ax2.set_ylabel("Number of Tasks", color="#FF8C00")
    ax.set_title("Accuracy and Task Distribution by Context Length Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)

    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )

    for bar, count in zip(bars2, task_counts):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{count}",
            ha="center",
            va="bottom",
        )

    # Add legends
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("accuracy_and_distribution.png", dpi=300, bbox_inches="tight")
    logger.info("Combined chart saved as 'accuracy_and_distribution.png'")

# Saving scores to json
def save_results_to_json(results, category_stats, overall_accuracy):
    """Save detailed results to JSON file."""
    output_data = {
        "overall_accuracy": overall_accuracy,
        "total_tasks": len(results),
        "category_stats": category_stats,
        "detailed_results": results,
    }

    with open("test_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Detailed results saved to 'test_results.json'")


async def main():
    logger.info("Starting LongBench evaluation")
    tasks = prepare_longbench2()
    logger.info("Loaded %d tasks", len(tasks))

    sem = asyncio.Semaphore(20)

    async def bounded_task(i, t):
        async with sem:
            return await run_task(i, t)

    logger.info("Launching %d tasks with concurrency=%d", len(tasks), 20)
    results = await asyncio.gather(*(bounded_task(i, t) for i, t in enumerate(tasks)))

    scored = [r for r in results if "correct" in r]
    accuracy = sum(r["correct"] for r in scored) / len(scored) if scored else 0
    errors = len(results) - len(scored)
    logger.info(
        "Completed %d tasks; scored=%d; errors=%d", len(results), len(scored), errors
    )
    logger.info("Accuracy: %.2f%%", accuracy * 100)

    # Analyze by context length category
    category_stats = {}
    for result in results:
        if "context_category" in result:
            category = result["context_category"]
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "correct": 0,
                    "context_lengths": [],
                }

            category_stats[category]["total"] += 1
            if result.get("correct", False):
                category_stats[category]["correct"] += 1
            category_stats[category]["context_lengths"].append(
                result.get("context_length", 0)
            )

    # Calculate accuracy by category
    category_accuracy = {}
    for category, stats in category_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        category_accuracy[category] = accuracy
        avg_context_length = sum(stats["context_lengths"]) / len(
            stats["context_lengths"]
        )
        logger.info(
            "Category %s: %d tasks, %.2f%% accuracy, avg context length: %.0f chars",
            category,
            stats["total"],
            accuracy * 100,
            avg_context_length,
        )

    # Create visualization
    create_accuracy_chart(category_accuracy, category_stats)

    # Save detailed results to JSON
    save_results_to_json(results, category_stats, accuracy)

if __name__ == "__main__":
    asyncio.run(main())
