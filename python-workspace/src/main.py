import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.68.123:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4")

# Prompt user for question
print("=" * 60)
print("AI Assistant")
print("=" * 60)
message = input("\nEnter your question: ").strip()

if not message:
    print("No question provided. Exiting.")
    exit(0)

if USE_OLLAMA:
    # Use LiteLLM with Ollama - try with tools, fallback without if not supported
    from litellm import completion
    from strands_tools import calculator, current_time
    import json
    
    print(f"\nUsing Ollama at {OLLAMA_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"\nProcessing your question...")
    
    # Get the actual callable functions from the modules
    calculator_func = calculator.calculator
    current_time_func = current_time.current_time
    
    # Map of available tools
    available_tools = {
        "calculator": calculator_func,
        "current_time": current_time_func
    }
    
    # Define tools in OpenAI format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": calculator_func.__doc__ or "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "current_time",
                "description": current_time_func.__doc__ or "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]
    
    messages = [{"role": "user", "content": message}]
    max_iterations = 5
    
    # Try with tools first, fallback to no tools if model doesn't support them
    try:
        for iteration in range(max_iterations):
            response = completion(
                model=f"openai/{OLLAMA_MODEL}",
                messages=messages,
                api_base=OLLAMA_URL,
                api_key="not-needed",
                tools=tools,
                timeout=120
            )
            
            assistant_message = response.choices[0].message
            
            # Check if model wants to call a tool
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                # Add assistant message to history
                messages.append(assistant_message)
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  → Calling tool: {tool_name} with args: {tool_args}")
                    
                    # Execute the tool
                    if tool_name in available_tools:
                        try:
                            if tool_name == "calculator":
                                result = available_tools[tool_name](tool_args["expression"])
                            else:
                                result = available_tools[tool_name]()
                            tool_result = str(result)
                        except Exception as e:
                            tool_result = f"Error: {str(e)}"
                    else:
                        tool_result = f"Error: Tool {tool_name} not found"
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Continue loop to get final answer
                continue
            else:
                # No more tool calls, we have the final answer
                break
        
        print(f"\nQuestion: {message}")
        print(f"Answer: {response.choices[0].message.content}")
        
    except Exception as e:
        # If tools not supported, retry without them
        error_msg = str(e).lower()
        if "does not support tools" in error_msg or "tool" in error_msg:
            print("  (Note: Model doesn't support tools, running without them)")
            response = completion(
                model=f"openai/{OLLAMA_MODEL}",
                messages=[{"role": "user", "content": message}],
                api_base=OLLAMA_URL,
                api_key="not-needed",
                timeout=60
            )
            print(f"\nQuestion: {message}")
            print(f"Answer: {response.choices[0].message.content}")
        else:
            raise
    
else:
    # Use strands Agent with AWS Bedrock
    from strands import Agent
    from strands_tools import calculator, current_time
    
    print("\nUsing AWS Bedrock")
    print(f"\nProcessing your question...")
    
    agent = Agent(tools=[calculator, current_time])
    agent(message)
