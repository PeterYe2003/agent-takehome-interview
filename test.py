import os
import time
import logging
import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
import random
from agents import Agent, Runner
import matplotlib.pyplot as plt
import json

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("agents-test")

def get_context_length_category(context_length: int) -> str:
    """Categorize context length into Short, Medium, or Long."""
    if context_length <= 144000:
        return "Short (0-144k words)"
    elif context_length <= 512000:
        return "Medium (144k-512k words)"
    else:
        return "Long (512k-8M words)"

agent = Agent(
    model="gpt-4.1",
    name="baseline-agent",
    instructions="""You are a precise question-answering agent for multiple-choice questions. 

Your task:
1. Carefully read the provided context and question
2. Analyze all the given choices (A, B, C, D)
3. Select the single best answer based on the context
4. Respond with ONLY the letter of your chosen answer (A, B, C, or D)
5. Do not provide explanations, reasoning, or additional text

Example:
Question: What is the capital of France?
A) London
B) Paris  
C) Berlin
D) Madrid

Correct response: B

Remember: Your response must start with exactly one letter (A, B, C, or D)."""
)

runner = Runner()


def prepare_longbench2(n=None):
    if n is None:
        n = int(os.getenv("N", "10"))  # Default to 10 for testing
    logger.info("Preparing dataset sample: n=%d", n)
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    total = len(dataset)
    if n > total:
        logger.warning("Requested n=%d exceeds dataset size=%d; using n=%d", n, total, total)
        n = total
    indices = random.sample(range(total), n)
    sample = [dataset[i] for i in indices]
    logger.info("Prepared %d tasks from LongBench-v2", len(sample))
    return sample

async def run_task(i: int, task: dict):
    q = task["question"]
    context = task.get("context", "")
    
    # Track original context length and category
    original_context_length = len(context)
    context_category = get_context_length_category(original_context_length)
    
    # Truncate context if too long (limit to ~50k chars to stay under context window)
    max_context_len = 300000
    if len(context) > max_context_len:
        context = context[:max_context_len] + "... [truncated]"
        logger.warning("Task %s: context truncated from %d to %d chars", i, len(task.get("context", "")), len(context))
    
    input_prompt = f"""Context:
{context}

Question:
{q}

Choices:
A) {task.get('choice_A', '')}
B) {task.get('choice_B', '')}
C) {task.get('choice_C', '')}
D) {task.get('choice_D', '')}

Answer:"""
    
    # Check total input length (limit to 10MB as per API)
    max_input_len = 10 * 1024 * 1024  # 10MB
    if len(input_prompt) > max_input_len:
        logger.error("Task %s: input too long (%d chars), skipping", i, len(input_prompt))
        return {"id": i, "error": "Input too long"}

    start_time = time.perf_counter()
    logger.debug("Task %s: starting; prompt_len=%d", i, len(input_prompt))
    try:
        result = await runner.run(agent, input=input_prompt)
        output = result.final_output.strip()
        is_correct = (output.upper().startswith(task["answer"]))
        duration = time.perf_counter() - start_time
        logger.info("Task %s completed in %.2fs; correct=%s", i, duration, is_correct)
        return {
            "id": i,
            "question": q,
            "choices": {k: task[k] for k in ["choice_A","choice_B","choice_C","choice_D"]},
            "expected": task["answer"],
            "output": output,
            "correct": is_correct,
            "context_length": original_context_length,
            "context_category": context_category,
            "duration": duration,
        }
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.exception("Task %s failed after %.2fs: %s", i, duration, e)
        return {
            "id": i, 
            "error": str(e),
            "context_length": original_context_length,
            "context_category": context_category,
            "duration": duration,
            "correct": False,
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

    sem = asyncio.Semaphore(5)

    async def bounded_task(i, t):
        async with sem:
            return await run_task(i, t)

    logger.info("Launching %d tasks with concurrency=%d", len(tasks), 5)
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