import os
import time
import logging
import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
import random
from agents import Agent, Runner, function_tool
import matplotlib.pyplot as plt
import json
from tools import compress_context_for_question, summarize_long_text
import re
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("agents-test")

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)

def get_context_length_category(context_length: int) -> str:
    """Categorize context length into Short, Medium, or Long."""
    if context_length <= 144000:
        return "Short (0-144k chars)"
    elif context_length <= 512000:
        return "Medium (144k-512k chars)"
    else:
        return "Long (512k-8M chars)"


client = OpenAI()


@function_tool
def compress_context_tool(
    context: str,
    question: str,
    choices: str,
    chunk_size: int = 500,
    max_chars: int = 20000,
    top_k: int = None
) -> str:
    """Compress a long context by selecting the most relevant parts for the given question and choices."""
    choices_dict = json.loads(choices) if isinstance(choices, str) else choices
    result = compress_context_for_question(context, question, choices_dict, chunk_size, max_chars, top_k)
    return result["compressed"]

def _letter_only(s: str) -> str | None:
    """Extract just the letter (A, B, C, D) from the response."""
    m = re.match(r'^\s*([ABCD])\b', s.strip().upper())
    return m.group(1) if m else None

agent = Agent(
    name="baseline-agent",
    instructions="""You are a precise question-answering agent for multiple-choice questions. 

Your task:
1. Use compress_context_tool to extract the most relevant parts
3. Carefully read the context (compressed if applicable) and question
4. Analyze all the given choices (A, B, C, D)
5. Select the single best answer based on the context
6. Respond with ONLY the letter of your chosen answer (A, B, C, or D)
7. Do not provide explanations, reasoning, or additional text

For the compress_context_tool:
- Pass the context, question, and choices as a JSON string
- The tool will return a compressed version of the context focusing on relevant content
- Use the outputted context as your new context

Example:
Question: What is the capital of France?
A) London
B) Paris
C) Berlin
D) Madrid

Correct response: B

Remember: Your response must start with exactly one letter (A, B, C, or D).""",
    model="gpt-4.1",
    tools=[compress_context_tool]
)

runner = Runner()


def prepare_longbench2(n=None):
    if n is None:
        n = int(os.getenv("N", "10"))  # Default to 10 for testing
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
    q = task["question"]
    context = task.get("context", "") or ""
    choices = {
        "A": task.get('choice_A', ''),
        "B": task.get('choice_B', ''),
        "C": task.get('choice_C', ''),
        "D": task.get('choice_D', '')
    }

    original_context_length = len(context)
    context_category = get_context_length_category(original_context_length)

    # --- Pre-compress on the client BEFORE building the prompt ---

    budget_chars = 120000 * 4
    if len(context) > budget_chars:
        logger.warning("Compressing context for task %s", i)
        compressed = await summarize_long_text(context)
    else:
        compressed = context
    
    logger.warning("Compressed context length: %d", len(compressed))
    # if the compressed context length is less than 10000, save the original context to a file and the compressed context to a file
    if len(compressed) < 10000:
        with open(f"original_context_{i}.txt", "w") as f:
            f.write(context)
        with open(f"compressed_context_{i}.txt", "w") as f:
            f.write(compressed)
        # exit(0)

    # Build prompt with compressed context
    input_prompt = (
        f"Context:\n{compressed}\n\n"
        f"Question:\n{q}\n\n"
        "Choices:\n"
        f"A) {choices['A']}\n"
        f"B) {choices['B']}\n"
        f"C) {choices['C']}\n"
        f"D) {choices['D']}\n\n"
        "Answer:"
    )

    start_time = time.perf_counter()
    logger.debug(
        "Task %s: starting; prompt_len=%d chars (~%d tokens), original_ctx=%d chars",
        i, len(input_prompt), len(input_prompt)//4, original_context_length
    )

    result = await runner.run(agent, input=input_prompt)
    raw_output = (result.final_output or "").strip()
    letter = _letter_only(raw_output)
    is_correct = (letter == task["answer"])
    duration = time.perf_counter() - start_time
    logger.info("Task %s completed in %.2fs; correct=%s", i, duration, is_correct)
    return {
        "id": i,
        "question": q,
        "choices": choices,
        "expected": task["answer"],
        "output": letter,
        "raw_output": raw_output,
        "correct": is_correct,
        "context_length": original_context_length,
        "context_category": context_category,
        "duration": duration,
    }


def create_accuracy_chart(category_accuracy, category_stats):
    """Create a bar chart showing accuracy by context length category."""
    categories = list(category_accuracy.keys())
    accuracies = [category_accuracy[cat] * 100 for cat in categories]
    task_counts = [category_stats[cat]["total"] for cat in categories]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy chart
    bars1 = ax1.bar(categories, accuracies, color=['#2E8B57', '#FF8C00', '#DC143C'])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy by Context Length Category')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Task count chart
    bars2 = ax2.bar(categories, task_counts, color=['#2E8B57', '#FF8C00', '#DC143C'])
    ax2.set_ylabel('Number of Tasks')
    ax2.set_title('Task Distribution by Context Length Category')
    
    # Add value labels on bars
    for bar, count in zip(bars2, task_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('accuracy_by_context_length.png', dpi=300, bbox_inches='tight')
    logger.info("Chart saved as 'accuracy_by_context_length.png'")
    
    # Also create a combined chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars with accuracy and task count
    x = range(len(categories))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy (%)', color='#2E8B57')
    ax2 = ax.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], task_counts, width, label='Task Count', color='#FF8C00')
    
    ax.set_xlabel('Context Length Category')
    ax.set_ylabel('Accuracy (%)', color='#2E8B57')
    ax2.set_ylabel('Number of Tasks', color='#FF8C00')
    ax.set_title('Accuracy and Task Distribution by Context Length Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    for bar, count in zip(bars2, task_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('accuracy_and_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Combined chart saved as 'accuracy_and_distribution.png'")

def save_results_to_json(results, category_stats, overall_accuracy):
    """Save detailed results to JSON file."""
    output_data = {
        "overall_accuracy": overall_accuracy,
        "total_tasks": len(results),
        "category_stats": category_stats,
        "detailed_results": results
    }
    
    with open('test_results.json', 'w') as f:
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
    logger.info("Completed %d tasks; scored=%d; errors=%d", len(results), len(scored), errors)
    logger.info("Accuracy: %.2f%%", accuracy * 100)
    
    # Analyze by context length category
    category_stats = {}
    for result in results:
        if "context_category" in result:
            category = result["context_category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0, "context_lengths": []}
            
            category_stats[category]["total"] += 1
            if result.get("correct", False):
                category_stats[category]["correct"] += 1
            category_stats[category]["context_lengths"].append(result.get("context_length", 0))
    
    # Calculate accuracy by category
    category_accuracy = {}
    for category, stats in category_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        category_accuracy[category] = accuracy
        avg_context_length = sum(stats["context_lengths"]) / len(stats["context_lengths"])
        logger.info("Category %s: %d tasks, %.2f%% accuracy, avg context length: %.0f chars", 
                   category, stats["total"], accuracy * 100, avg_context_length)
    
    # Create visualization
    create_accuracy_chart(category_accuracy, category_stats)
    
    # Save detailed results to JSON
    save_results_to_json(results, category_stats, accuracy)

if __name__ == "__main__":
    asyncio.run(main())