import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
USE_LM_STUDIO = os.getenv("USE_LM_STUDIO", "false").lower() == "true"
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://192.168.68.123:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "phi-4")

if USE_LM_STUDIO:
    # Use LiteLLM with LM Studio's OpenAI-compatible endpoint
    from litellm import completion
    
    print(f"Using LM Studio at {LM_STUDIO_URL}")
    print(f"Model: {LM_STUDIO_MODEL}")
    
    message = "I was born in 1979, what is my current age?"
    
    response = completion(
        model=f"openai/{LM_STUDIO_MODEL}",
        messages=[{"role": "user", "content": message}],
        api_base=LM_STUDIO_URL,
        api_key="not-needed"  # LM Studio doesn't require an API key
    )
    
    print(f"\nQuestion: {message}")
    print(f"Answer: {response.choices[0].message.content}")
    
else:
    # Use strands Agent with AWS Bedrock
    from strands import Agent
    from strands_tools import calculator, current_time
    
    print("Using AWS Bedrock")
    
    agent = Agent(tools=[calculator, current_time])
    message = "I was born in 1979, what is my current age?"
    agent(message)
