import os
import streamlit as st
from dotenv import load_dotenv
from litellm import completion
import json

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.68.123:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model info
    st.info(f"**Model**: {OLLAMA_MODEL}\n\n**Endpoint**: {OLLAMA_URL}")
    
    # Reasoning toggle
    st.session_state.show_reasoning = st.checkbox(
        "Show reasoning steps",
        value=st.session_state.show_reasoning,
        help="Display tool calls and intermediate reasoning steps"
    )
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Powered by Ollama + Strands Tools")

# Main title
st.title("🤖 AI Assistant")
st.caption("Chat with AI - powered by " + ("Ollama" if USE_OLLAMA else "AWS Bedrock"))

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show reasoning if enabled and available
        if st.session_state.show_reasoning and "reasoning" in message:
            with st.expander("🔍 Reasoning Steps"):
                for step in message["reasoning"]:
                    st.json(step)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        reasoning_placeholder = st.expander("🔍 Reasoning Steps") if st.session_state.show_reasoning else None
        
        try:
            if USE_OLLAMA:
                # Import tools
                from strands_tools import calculator, current_time
                from strands_tools.tavily import tavily_search
                import asyncio
                
                # Get callable functions
                calculator_func = calculator.calculator
                current_time_func = current_time.current_time
                
                # Create sync wrapper for async tavily_search
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
                
                # Conversation messages
                messages = [{"role": msg["role"], "content": msg["content"]} 
                           for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]]
                
                reasoning_steps = []
                max_iterations = 5
                
                # Tool calling loop
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
                                
                                # Record reasoning step
                                reasoning_step = {
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "arguments": tool_args
                                }
                                
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
                                            
                                            # Format for display
                                            if isinstance(result, dict) and result.get("status") == "success":
                                                formatted_parts = []
                                                
                                                # Include AI-generated answer if available
                                                if include_answer and "answer" in result:
                                                    formatted_parts.append(f"**AI Summary:** {result['answer']}\n")
                                                
                                                # Extract source content
                                                content = result.get("content", [])
                                                if content and isinstance(content, list):
                                                    formatted_parts.append("**Sources:**")
                                                    for idx, item in enumerate(content[:max_results], 1):
                                                        if isinstance(item, dict):
                                                            title = item.get("title", "No title")
                                                            url = item.get("url", "")
                                                            snippet = item.get("text", "")[:200]
                                                            formatted_parts.append(f"{idx}. [{title}]({url})")
                                                            formatted_parts.append(f"   {snippet}...")
                                                    
                                                    result = "\n".join(formatted_parts)
                                        else:
                                            result = available_tools[tool_name]()
                                        tool_result = str(result)
                                        reasoning_step["result"] = tool_result
                                    except Exception as e:
                                        tool_result = f"Error: {str(e)}"
                                        reasoning_step["error"] = str(e)
                                else:
                                    tool_result = f"Error: Tool {tool_name} not found"
                                    reasoning_step["error"] = f"Tool {tool_name} not found"
                                
                                reasoning_steps.append(reasoning_step)
                                
                                # Show reasoning in real-time if enabled
                                if reasoning_placeholder:
                                    reasoning_placeholder.json(reasoning_steps)
                                
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
                    
                    final_answer = response.choices[0].message.content
                    message_placeholder.markdown(final_answer)
                    
                    # Save assistant response with reasoning
                    assistant_msg = {"role": "assistant", "content": final_answer}
                    if reasoning_steps:
                        assistant_msg["reasoning"] = reasoning_steps
                    st.session_state.messages.append(assistant_msg)
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "does not support tools" in error_msg or "tool" in error_msg:
                        # Fallback to no tools
                        response = completion(
                            model=f"openai/{OLLAMA_MODEL}",
                            messages=[{"role": msg["role"], "content": msg["content"]} 
                                     for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]],
                            api_base=OLLAMA_URL,
                            api_key="not-needed",
                            timeout=120
                        )
                        final_answer = response.choices[0].message.content
                        message_placeholder.markdown(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    else:
                        raise
            else:
                # AWS Bedrock path
                from strands import Agent
                from strands_tools import calculator, current_time
                
                agent = Agent(tools=[calculator, current_time])
                result = agent(prompt)
                
                message_placeholder.markdown(str(result))
                st.session_state.messages.append({"role": "assistant", "content": str(result)})
        
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
