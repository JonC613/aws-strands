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
    from strands_tools.tavily import tavily_search
    import json
    import asyncio
    
    print(f"\nUsing Ollama at {OLLAMA_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"\nProcessing your question...")
    
    # Get the actual callable functions from the modules
    calculator_func = calculator.calculator
    current_time_func = current_time.current_time
    
    # Create sync wrapper for async tavily_search with enhanced parameters
    def tavily_search_sync(query: str, search_depth: str = "basic", max_results: int = 5, include_answer: bool = True) -> dict:
        """
        Synchronous wrapper for async tavily_search with enhanced parameters.
        
        Args:
            query: Search query string
            search_depth: "basic" (1 credit) or "advanced" (2 credits) - advanced provides better relevance
            max_results: Number of results to return (1-10 recommended)
            include_answer: Include AI-generated answer summary
        """
        return asyncio.run(tavily_search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=include_answer
        ))
    
    # Map of available tools
    available_tools = {
        "calculator": calculator_func,
        "current_time": current_time_func,
        "tavily_search": tavily_search_sync
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
        },
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "Search the web for real-time, up-to-date information using Tavily's AI-optimized search engine. Use this for current events, recent news, live data, or any information that changes over time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Be specific and clear. Examples: 'latest SpaceX launch date', 'current Bitcoin price', 'recent AI breakthroughs 2025'"
                        },
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": "Search depth: 'basic' for quick results, 'advanced' for more thorough and relevant results. Default: 'basic'"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of search results to return (1-10). More results = more context but longer processing. Default: 5",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "include_answer": {
                            "type": "boolean",
                            "description": "Whether to include an AI-generated summary answer from Tavily. Recommended: true. Default: true"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [{"role": "user", "content": message}]
    max_iterations = 5
    
    # Try with tools first, fallback to no tools if model doesn't support them
    try:
        for iteration in range(max_iterations):
            print(f"  [Iteration {iteration + 1}/{max_iterations}]")
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
                            elif tool_name == "tavily_search":
                                # Extract optional parameters with defaults
                                search_depth = tool_args.get("search_depth", "basic")
                                max_results = tool_args.get("max_results", 5)
                                include_answer = tool_args.get("include_answer", True)
                                
                                result = available_tools[tool_name](
                                    query=tool_args["query"],
                                    search_depth=search_depth,
                                    max_results=max_results,
                                    include_answer=include_answer
                                )
                                
                                # Format Tavily results for the model
                                if isinstance(result, dict) and result.get("status") == "success":
                                    formatted_parts = []
                                    
                                    # Include AI-generated answer if available (most important!)
                                    if include_answer and "answer" in result:
                                        formatted_parts.append(f"AI Summary: {result['answer']}\n")
                                    
                                    # Extract source content
                                    content = result.get("content", [])
                                    if content and isinstance(content, list):
                                        formatted_parts.append("Sources:")
                                        for idx, item in enumerate(content[:max_results], 1):
                                            if isinstance(item, dict):
                                                title = item.get("title", "No title")
                                                url = item.get("url", "")
                                                snippet = item.get("text", "")[:200]
                                                formatted_parts.append(f"{idx}. [{title}]({url})\n   {snippet}...")
                                        
                                        result = "\n".join(formatted_parts) if formatted_parts else "No results found"
                                    else:
                                        result = "\n".join(formatted_parts) if formatted_parts else str(content)[:500]
                                elif isinstance(result, dict) and result.get("status") == "error":
                                    result = f"Search error: {result.get('content', [{}])[0].get('text', 'Unknown error')}"
                                else:
                                    result = str(result)[:500]
                            else:
                                result = available_tools[tool_name]()
                            tool_result = str(result)
                        except Exception as e:
                            tool_result = f"Error: {str(e)}"
                    else:
                        tool_result = f"Error: Tool {tool_name} not found"                    # Add tool result to messages
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
        
        # Check if we have a final answer
        final_content = response.choices[0].message.content
        if final_content:
            print(f"\nQuestion: {message}")
            print(f"Answer: {final_content}")
        else:
            # Reached max iterations, force a final answer WITHOUT tools
            print(f"\n  [Max iterations reached, requesting final answer...]")
            messages.append({
                "role": "user",
                "content": "Based on the information you gathered, please provide a concise final answer to my original question. Do not use any more tools."
            })
            print(f"  [Calling model for final answer...]")
            response = completion(
                model=f"openai/{OLLAMA_MODEL}",
                messages=messages,
                api_base=OLLAMA_URL,
                api_key="not-needed",
                timeout=60
                # Don't include tools parameter to prevent more tool calls
            )
            print(f"  [Got response]")
            print(f"\nQuestion: {message}")
            final_answer = response.choices[0].message.content
            if final_answer:
                print(f"Answer: {final_answer}")
            else:
                print(f"Answer: Unable to generate final answer.")
        
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
