import os
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv


load_dotenv()


@function_tool
def get_weather(city: str) -> str:
	"""Returns weather info for the specified city."""
	return f"The weather in {city} is sunny."


def create_agent() -> Agent:
	return Agent(
		name="Haiku Agent",
		instructions="Always respond in haiku form.",
		model="o3-mini",
		tools=[get_weather],
	)


def main() -> None:
	# Ensure API key is present
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError(
			"OPENAI_API_KEY not set. Create a .env with OPENAI_API_KEY=... or export it."
		)

	agent = create_agent()
	prompt = "What's the weather in Tokyo?"
	result = Runner.run_sync(agent, prompt)
	print(result.final_output)


if __name__ == "__main__":
	main()