# Basic OpenAI Agents SDK Example

This repo contains a minimal Python example using the OpenAI Agents SDK to create a basic agent with a simple tool and run it synchronously.

## Prerequisites
- Python 3.9+
- An OpenAI API key with access to your chosen model

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

3. Set your API key. Either export it:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
   Or create a `.env` file in the repo root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Run
```bash
python main.py
```

The script will:
- Define a `get_weather` tool via `@function_tool`
- Create a simple agent with instructions and model (`o3-mini` by default)
- Run the agent synchronously with a sample prompt

If you need a different model, change the `model` field in `create_agent()` in `main.py`.

## Reference
- OpenAI Agents SDK docs: [Agents](https://openai.github.io/openai-agents-python/agents/)
