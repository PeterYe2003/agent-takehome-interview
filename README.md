# Agent Takehome Interview Project

This repo contains a comprehensive evaluation of AI agents for long-context question answering, including both baseline implementations and evolutionary approaches to tool optimization.

## Project Overview

This project evaluates AI agents on the LongBench-v2 dataset for long-context question answering. It includes:

1. **Baseline Agent Implementation** (`test.py`) - An agent with keyword extraction and evidence search tools
2. **Evolutionary Tool Optimization** (`evolution.py`) - Genetic algorithm approach to improve search tools using AI-generated code mutations
## Prerequisites
- Python 3.12+
- OpenAI API key
- Anthropic API key (for evolution experiments)

## Setup
1. Create and activate a virtual environment (recommended):
   - macOS/Linux:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```bash
     python -m venv .venv; .venv\Scripts\Activate.ps1
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys. Either export them:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export ANTHROPIC_API_KEY=your_anthropic_key_here
   ```
   Or create a `.env` file in the repo root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

## Usage

### Run the Baseline Agent Evaluation
```bash
python test.py
```

This will:
- Load the LongBench-v2 dataset
- Run the agent on 100 sample tasks (configurable via `N` environment variable)
- Generate performance metrics and visualizations
- Save results to `test_results.json` and charts

### Run the Evolution Experiment
```bash
python evolution.py
```

This will:
- Start with a simple search tool
- Use AI-generated mutations to evolve better search algorithms
- Run for 3 generations with 5 variants each
- Evaluate each variant on a benchmark

## Key Findings

### Baseline Agent Performance
- **Overall Accuracy**: ~41% (significant improvement over simple approaches)
- **Context Length Performance**:
  - Short contexts (0-144k chars): ~40% accuracy
  - Medium contexts (144k-512k chars): ~42% accuracy  
  - Long contexts (512k-8M chars): ~45% accuracy
- **Key Innovation**: Combined keyword extraction and relevant text finding tool

### Evolution Experiment Limitations

The evolutionary approach revealed several critical limitations in AI-generated code:

#### 1. SequenceMatcher API Misuse
**Problem**: AI consistently generates code that calls `.lower()` on `SequenceMatcher` objects:
```python
# ❌ WRONG - AI generates this
sm = SequenceMatcher(None, query.lower(), doc.lower())
result = sm.lower()  # AttributeError: 'SequenceMatcher' object has no attribute 'lower'

# ✅ CORRECT - Should be
sm = SequenceMatcher(None, query.lower(), doc.lower())
result = sm.ratio()  # Get similarity ratio
```

**Impact**: Most evolved variants crash with `'SequenceMatcher' object has no attribute 'lower'` errors.

#### 2. Counter Initialization Errors
**Problem**: AI generates incorrect `Counter` usage:
```python
# ❌ WRONG - AI generates this
Counter(query_tokens, doc_tokens)  # TypeError: takes from 1 to 2 positional arguments but 3 were given

# ✅ CORRECT - Should be
Counter(query_tokens)  # or Counter(doc_tokens)
```

**Impact**: Additional variants in evolution.py crash with `Counter.__init__()` errors.

#### 5. Context Window Limitations (evolution_baseline_combined.py)
**Problem**: The more complex evolution system faces severe context window issues:
- **Context Length Exceeded**: `Error code: 400 - Your input exceeds the context window of this model`
- **Maximum Turns Exceeded**: `Max turns (10) exceeded` - Agent conversations hit the turn limit before completing

**Impact**: The combined evolution system cannot evaluate search functions on real LongBench data, making it impossible to evolve better search algorithms.

**Example Errors**:
```
Error code: 400 - {'error': {'message': 'Your input exceeds the context window of this model. Please adjust your input and try again.', 'type': 'invalid_request_error', 'param': 'input', 'code': 'context_length_exceeded'}}
WARNING: Search function evaluation failed on item 2: Error code: 400 - Your input exceeds the context window of this model
WARNING: Search function evaluation failed on item 1: Max turns (10) exceeded
```

**Interesting Finding**: Despite the failures, the evolution process generates interesting penalization functions that penalize lengthy responses, suggesting the AI is learning to optimize for conciseness even when the main evaluation fails.

## Files

- `test.py` - Main agent evaluation script
- `evolution.py` - Evolutionary tool optimization experiment
- `longbench_evolution.py` - Alternative evolution implementation
- `main.py` - Basic OpenAI Agents SDK example
- `tools.py` - Utility functions
- `requirements.txt` - Python dependencies

## Reference
- OpenAI Agents SDK docs: [Agents](https://openai.github.io/openai-agents-python/agents/)
- LongBench-v2 Dataset: [Paper](https://arxiv.org/abs/2406.14805)
