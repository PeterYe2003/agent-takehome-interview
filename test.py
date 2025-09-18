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
# No longer using tools from tools.py - using local context with search tools instead
import re
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Any

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


@function_tool
def extract_question_keywords_tool(
    wrapper: RunContextWrapper[TaskContext]
) -> str:
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
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
        logger.warning(f"Failed to extract question keywords for task {task_context.task_id}: {e}")
        return "Failed to extract question keywords"

@function_tool
def extract_option_keywords_tool(
    wrapper: RunContextWrapper[TaskContext],
    option_letter: str
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        keywords = response.choices[0].message.content.strip()
        
        # Append to combined keywords file for debugging
        with open(f"keywords_task_{task_context.task_id}.txt", "a") as f:
            f.write(f"Option {option_letter} Keywords: {keywords}\n")
            
        return f"Option {option_letter} keywords: {keywords}"
    except Exception as e:
        logger.warning(f"Failed to extract option {option_letter} keywords for task {task_context.task_id}: {e}")
        return f"Failed to extract option {option_letter} keywords"


@function_tool
def analyze_question_type_tool(
    wrapper: RunContextWrapper[TaskContext]
) -> str:
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=150,
            temperature=0.1
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
    except Exception as e:
        logger.warning(f"Failed to analyze question type for task {task_context.task_id}: {e}")
        return f"Context Length: {context_length:,} characters, Category: {context_category}. Use standard parameters."

@function_tool
def get_full_context_tool(
    wrapper: RunContextWrapper[TaskContext]
) -> str:
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
        logger.warning(f"Failed to save full context for task {task_context.task_id}: {e}")
    
    return f"Full context ({len(context):,} characters):\n\n{context}"

@function_tool
def search_evidence_for_option_tool(
    wrapper: RunContextWrapper[TaskContext],
    option_letter: str,
    keywords: str,
    expansion_chars: int = 500,
    max_results: int = 15
) -> str:
    """Search the context for evidence supporting a specific option using the provided keywords."""
    task_context = wrapper.context
    context = task_context.context

    if option_letter not in task_context.choices:
        return f"Invalid option letter: {option_letter}"

    option_text = task_context.choices[option_letter]

    # Parse keywords from the string (comma-separated)
    keyword_list = [kw.strip().lower() for kw in keywords.split(',') if kw.strip()]

    if not keyword_list:
        return f"No valid keywords provided for searching evidence for option {option_letter}."

    context_lower = context.lower()

    # Find all matches for each keyword with scoring
    scored_matches = []

    for keyword in keyword_list:
        if not keyword or len(keyword) < 2:  # Allow shorter keywords
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
                    score += 3  # Higher bonus for multiple keywords

            # Bonus for option-specific terms appearing in context
            option_words = re.findall(r'\b\w+\b', option_text.lower())
            for ow in option_words:
                if len(ow) > 3 and ow in text_lower:
                    score += 2

            # Bonus for question-related terms
            question_words = re.findall(r'\b\w+\b', task_context.question.lower())
            for qw in question_words:
                if len(qw) > 3 and qw in text_lower:
                    score += 1

            # Penalty for very common words (but less penalty)
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']
            for cw in common_words:
                if cw in text_lower:
                    score -= 0.05

            scored_matches.append({
                'keyword': keyword,
                'position': pos,
                'text': expanded_text.strip(),
                'score': score,
                'keywords_found': [kw for kw in keyword_list if kw in text_lower],
                'option_terms_found': [ow for ow in option_words if len(ow) > 3 and ow in text_lower]
            })

            start = pos + 1

    if not scored_matches:
        return f"No evidence found for option {option_letter} with keywords: {', '.join(keyword_list)}"

    # Sort by score (highest first), then by position
    scored_matches.sort(key=lambda x: (-x['score'], x['position']))

    # Remove duplicates and select top results
    unique_texts = []
    seen_texts = set()

    for match in scored_matches:
        # Use a normalized version for deduplication
        normalized_text = re.sub(r'\s+', ' ', match['text'].lower().strip())

        if normalized_text not in seen_texts and len(unique_texts) < max_results:
            keywords_found = ', '.join(match['keywords_found'])
            option_terms = ', '.join(match['option_terms_found'])
            unique_texts.append(f"[Score: {match['score']:.1f}, Keywords: {keywords_found}, Option terms: {option_terms}]: {match['text']}")
            seen_texts.add(normalized_text)

    if not unique_texts:
        return f"No unique evidence found for option {option_letter} with keywords: {', '.join(keyword_list)}"

    # Save search results to combined evidence file for debugging/analysis
    try:
        # Check if this is the first option being processed (create new file)
        # or if we're appending to existing file
        file_mode = "w" if option_letter == "A" else "a"
        
        with open(f"evidence_task_{task_context.task_id}.txt", file_mode) as f:
            if option_letter == "A":
                # Write header only for the first option
                f.write(f"Task ID: {task_context.task_id}\n")
                f.write(f"Question: {task_context.question}\n")
                f.write(f"Choices: {task_context.choices}\n")
                f.write(f"Answer: {task_context.answer}\n")
                f.write(f"Context Length: {len(context)} chars\n")
                f.write(f"Context Category: {task_context.context_category}\n\n")
            
            f.write(f"Evidence for Option {option_letter}: {option_text}\n")
            f.write(f"Keywords Searched: {', '.join(keyword_list)}\n")
            f.write(f"Expansion Chars: {expansion_chars}\n")
            f.write(f"Max Results: {max_results}\n")
            f.write(f"Total Matches Found: {len(scored_matches)}\n")
            f.write(f"Unique Results Returned: {len(unique_texts)}\n\n")
            f.write(f"Evidence for Option {option_letter} (by relevance score):\n")
            f.write("=" * 60 + "\n")
            for i, text in enumerate(unique_texts, 1):
                f.write(f"\nEvidence {i}:\n{text}\n")
                f.write("-" * 40 + "\n")
            f.write("\n\n")
    except Exception as e:
        logger.warning(f"Failed to save evidence for option {option_letter} task {task_context.task_id}: {e}")

    result = f"Evidence for Option {option_letter} ({len(unique_texts)} sections found):\n\n" + "\n\n".join(unique_texts)
    return result


def _letter_only(s: str) -> str | None:
    """Extract just the letter (A, B, C, D) from the response."""
    m = re.match(r'^\s*([ABCD])\b', s.strip().upper())
    return m.group(1) if m else None


agent = Agent[TaskContext](
    name="systematic-evidence-agent",
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
- Use 60-98% of available context based on length
- Prioritize max_results over expansion_chars for broader coverage
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
    tools=[analyze_question_type_tool, get_full_context_tool, extract_question_keywords_tool, extract_option_keywords_tool, search_evidence_for_option_tool]
)

runner = Runner()


def prepare_longbench2(n=None):
    if n is None:
        n = int(os.getenv("N", "50"))  # Default to 50 for testing
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

    # Create the task context object with the full context
    # The search tools will handle finding relevant sections
    task_context = TaskContext(
        question=q,
        context=context,  # Use the full context - let the search tools handle filtering
        choices=choices,
        answer=task["answer"],
        task_id=i,
        original_context_length=original_context_length,
        context_category=context_category
    )

    # Build a simple prompt that doesn't include the context (it's now in local context)
    input_prompt = (
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

    result = await runner.run(agent, input=input_prompt, context=task_context)
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
