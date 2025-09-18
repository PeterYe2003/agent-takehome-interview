import os
import time
import logging
import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
import random
from agents import Agent, Runner

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("agents-test")


agent = Agent(
    model="gpt-4o",
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
    
    # Truncate context if too long (limit to ~50k chars to stay under context window)
    max_context_len = 100000
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
        }
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.exception("Task %s failed after %.2fs: %s", i, duration, e)
        return {"id": i, "error": str(e)}


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

if __name__ == "__main__":
    asyncio.run(main())