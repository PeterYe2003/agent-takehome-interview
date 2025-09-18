#!/usr/bin/env python3
"""
Simple test script to verify the context refactoring works correctly.
This creates a minimal test case to ensure the local context approach functions properly.
"""

import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, RunContextWrapper
from dataclasses import dataclass
from typing import Dict, Any
import json
# No longer using tools from tools.py

load_dotenv()

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
def extract_keywords_tool(
    wrapper: RunContextWrapper[TaskContext]
) -> str:
    """Extract important keywords from the question and answer choices to help identify relevant context."""
    task_context = wrapper.context
    
    # Combine question and all choices
    full_text = f"{task_context.question} {' '.join(task_context.choices.values())}"
    
    # Extract keywords using simple heuristics
    # Remove common stop words and extract meaningful terms
    stop_words = {
        "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "by", "with", "is", "are", 
        "was", "were", "be", "as", "at", "from", "that", "this", "it", "its", "into", "than", 
        "then", "there", "their", "them", "these", "those", "about", "also", "can", "could", 
        "should", "will", "would", "may", "might", "must", "what", "which", "who", "when", 
        "where", "why", "how", "all", "any", "some", "each", "every", "other", "another", 
        "such", "only", "just", "more", "most", "less", "least", "much", "many", "few", 
        "little", "big", "small", "good", "bad", "new", "old", "first", "last", "next", 
        "previous", "same", "different", "same", "true", "false", "yes", "no"
    }
    
    # Extract words (alphanumeric + hyphens)
    import re
    words = re.findall(r"[A-Za-z0-9\-']+", full_text.lower())
    
    # Filter out stop words and short words, keep only meaningful terms
    keywords = []
    for word in words:
        if (len(word) > 2 and 
            word not in stop_words and 
            not word.isdigit() and
            word not in keywords):  # Avoid duplicates
            keywords.append(word)
    
    # Sort by length (longer words often more specific) and frequency
    word_counts = {}
    for word in words:
        if word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency first, then by length
    keywords.sort(key=lambda x: (-word_counts[x], -len(x)))
    
    # Return top keywords (limit to most important ones)
    top_keywords = keywords[:8]  # Reduced to top 8 for better search results
    
    result = f"Top keywords from question and choices: {', '.join(top_keywords)}"
    return result


@function_tool
def search_context_tool(
    wrapper: RunContextWrapper[TaskContext],
    keywords: str,
    expansion_chars: int = 400
) -> str:
    """Search the context for the given keywords and return expanded text around matches."""
    task_context = wrapper.context
    context = task_context.context
    
    # Parse keywords from the string (comma-separated)
    keyword_list = [kw.strip().lower() for kw in keywords.split(',')]
    
    # Find all matches for each keyword in the context
    matches = []
    context_lower = context.lower()
    
    for keyword in keyword_list:
        if not keyword:
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
            
            # Add some context markers
            if expand_start > 0:
                expanded_text = "..." + expanded_text
            if expand_end < len(context):
                expanded_text = expanded_text + "..."
            
            matches.append({
                'keyword': keyword,
                'position': pos,
                'text': expanded_text.strip()
            })
            
            start = pos + 1
    
    # Sort matches by position in the original context
    matches.sort(key=lambda x: x['position'])
    
    # Combine all unique matches
    unique_texts = []
    seen_texts = set()
    
    for match in matches:
        if match['text'] not in seen_texts:
            unique_texts.append(f"[Found '{match['keyword']}']: {match['text']}")
            seen_texts.add(match['text'])
    
    if not unique_texts:
        return f"No matches found for keywords: {', '.join(keyword_list)}"
    
    result = f"Relevant context sections found:\n\n" + "\n\n".join(unique_texts)
    return result

agent = Agent[TaskContext](
    name="test-agent",
    instructions="""You are a precise question-answering agent for multiple-choice questions. 

Your task:
1. FIRST: Use extract_keywords_tool to identify the top 8 most important keywords from the question and choices
2. SECOND: Use search_context_tool with those keywords to find relevant sections of the context
3. Carefully analyze the found context sections and the question
4. Analyze all the given choices (A, B, C, D)
5. Select the single best answer based on the relevant context sections
6. Respond with ONLY the letter of your chosen answer (A, B, C, or D)
7. Do not provide explanations, reasoning, or additional text

Tool Usage:
- extract_keywords_tool: Automatically extracts top keywords from question and choices
- search_context_tool: Takes the keywords (as a comma-separated string) and searches the context for relevant sections with 400-character expansion around matches

Workflow:
1. Call extract_keywords_tool to get keywords
2. Copy the keywords from the result and call search_context_tool with them
3. Use the found context sections to answer the question

Remember: Your response must start with exactly one letter (A, B, C, or D).""",
    model="gpt-4.1",
    tools=[extract_keywords_tool, search_context_tool]
)

async def test_context_refactor():
    """Test the refactored context approach with a simple example."""
    
    # Create a simple test case
    test_context = """
    The capital of France is Paris. Paris is located in the north-central part of France.
    It is the largest city in France and serves as the country's political and economic center.
    Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum,
    and the Notre-Dame Cathedral. The city has a rich history dating back to ancient times.
    """
    
    test_question = "What is the capital of France?"
    test_choices = {
        "A": "London",
        "B": "Paris", 
        "C": "Berlin",
        "D": "Madrid"
    }
    
    # Create the task context
    task_context = TaskContext(
        question=test_question,
        context=test_context,
        choices=test_choices,
        answer="B",
        task_id=1,
        original_context_length=len(test_context),
        context_category="Short (0-144k chars)"
    )
    
    # Simple prompt without context (context is now in local context)
    input_prompt = f"""Question:
{test_question}

Choices:
A) {test_choices['A']}
B) {test_choices['B']}
C) {test_choices['C']}
D) {test_choices['D']}

Answer:"""
    
    print("Testing context refactoring...")
    print(f"Question: {test_question}")
    print(f"Expected answer: {task_context.answer}")
    print(f"Context length: {len(test_context)} characters")
    print("Running agent with local context...")
    
    try:
        runner = Runner()
        result = await runner.run(agent, input=input_prompt, context=task_context)
        
        print(f"Agent response: {result.final_output}")
        
        # Extract the letter from the response
        import re
        m = re.match(r'^\s*([ABCD])\b', result.final_output.strip().upper())
        letter = m.group(1) if m else None
        
        print(f"Extracted answer: {letter}")
        print(f"Correct: {letter == task_context.answer}")
        
        if letter == task_context.answer:
            print("✅ Test passed! Context refactoring works correctly.")
        else:
            print("❌ Test failed. Agent did not provide the expected answer.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_context_refactor())
