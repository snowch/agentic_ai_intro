#!/usr/bin/env python3

import json
import logging
from typing import TypedDict, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define our state type
class AgentState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]
    next: str

# Define our hello tool
@tool
def say_hello(name: str) -> str:
    """Says hello to the provided name."""
    return f"Hello, {name}!"

# Create our model
llm = OllamaLLM(model="internlm2:1.8b-chat-v2.5-q2_K")

def call_model(state: AgentState) -> AgentState:
    """Call the model to get the next action."""
    messages = state['messages']
    # Use an f-string instead of .format() to avoid issues with JSON examples
    prompt = f"""You are a helpful assistant that MUST respond with ONLY valid JSON.

You have access to this tool:
say_hello: A tool that says hello to a name

EXAMPLES:

User: Please say hello to Bob
Response: {{"tool": "say_hello", "tool_input": "Bob"}}

User: What's the weather?
Response: {{"final_answer": "I apologize, but I don't have access to weather information."}}

RULES:
1. Respond with ONLY JSON
2. No explanations or extra text
3. Use EXACTLY one of these formats:
   {{"tool": "say_hello", "tool_input": "<name>"}}
   {{"final_answer": "<your response>"}}

Current request: {messages[-1].content}

JSON response:"""

    logging.debug("Prompt sent to LLM: %s", prompt)

    try:
        response = llm.invoke(prompt).strip()
        logging.debug("Raw LLM response: %s", response)

        # Try to extract JSON from the response by finding the first { and last }
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            parsed_response = json.loads(json_str)

            if "tool" in parsed_response:
                state['next'] = 'call_tool'
            else:
                state['next'] = END

            state['messages'].append(AIMessage(content=json.dumps(parsed_response)))
        else:
            raise ValueError("No JSON object found in response")

    except Exception as e:
        logging.error("Error processing LLM response: %s", e)
        state['messages'].append(AIMessage(content=json.dumps({
            "final_answer": "I encountered an error while processing your request."
        })))
        state['next'] = END

    return state

def call_tool(state: AgentState) -> AgentState:
    """Execute the tool based on the model's decision."""
    last_message = state['messages'][-1]

    try:
        parsed_content = json.loads(last_message.content)
        if "tool" in parsed_content and parsed_content["tool"] == "say_hello":
            tool_input = parsed_content.get("tool_input", "")
            if tool_input:
                result = say_hello(tool_input)
                state['messages'].append(AIMessage(content=json.dumps({
                    "final_answer": result
                })))
            else:
                state['messages'].append(AIMessage(content=json.dumps({
                    "final_answer": "No name was provided to say hello to."
                })))
        else:
            state['messages'].append(AIMessage(content=json.dumps({
                "final_answer": "I couldn't execute the tool properly."
            })))
    except Exception as e:
        logging.error("Error executing tool: %s", e)
        state['messages'].append(AIMessage(content=json.dumps({
            "final_answer": f"Error executing tool: {str(e)}"
        })))

    state['next'] = END
    return state

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_model", call_model)
workflow.add_node("call_tool", call_tool)

# Add edges
workflow.add_edge("call_model", "call_tool")
workflow.add_edge("call_model", END)
workflow.add_edge("call_tool", END)

# Set entry point
workflow.set_entry_point("call_model")

# Compile the graph
chain = workflow.compile()

if __name__ == "__main__":
    try:
        result = chain.invoke({
            "messages": [HumanMessage(content="Please say hello to Alice")],
            "next": ""
        })
        # Get the final message
        final_message = result["messages"][-1].content
        try:
            # Try to parse it as JSON
            parsed_message = json.loads(final_message)
            if "final_answer" in parsed_message:
                print(parsed_message["final_answer"])
            else:
                print(parsed_message)
        except json.JSONDecodeError:
            # If it's not valid JSON, just print the message as is
            print(final_message)
    except Exception as e:
        logging.critical("Critical error occurred: %s", e, exc_info=True)
        print(f"An error occurred: {str(e)}")
