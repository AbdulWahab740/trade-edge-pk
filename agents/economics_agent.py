# agents/economics_agent.py
import os
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel,Field
from typing import Literal
import json 

# Use your existing Cohere API key
os.environ["COHERE_API_KEY"] = "oM4BkpsY3NEdlFxktdmYEtOHPJ7TBHJD5WcvvC2e"

class EconomicsResponse(BaseModel):
    output: str
    insights_type: Literal["growth", "opportunity", "trend", "plan",'recommendation',"comparison"] = Field(description="Type of economic insights provided (e.g., growth, opportunity, trend, plan, recommendation, comparison)")
    insights: str = Field(description="Short economic insights based on the provided data")
# Define the state schema
class EconomicsAgentState(TypedDict):
    messages: List
    tools: List
    input: str
    agent_outcome: Dict[str, Any]
    agent_scratchpad: str

# Initialize LLM with Cohere
llm = ChatCohere(
    model="command-r7b-12-2024",
    temperature=0.3
)
def economics_agent_node(state: EconomicsAgentState) -> EconomicsAgentState:    
    """Process input using economics agent and enforce structured JSON output."""
    
    # 1. ENFORCE STRUCTURED OUTPUT
    structured_llm = llm.with_structured_output(
        schema=EconomicsResponse, 
        include_raw=False
    )
    
    prompt_template = PromptTemplate(
        input_variables=["input", "context", "history"],
        template="""You are an Economics Analyst for Pakistan's trade data (2025).

Your expertise:
- Economic policy analysis
- International trade comparisons
- Strategic recommendations
- Market trends and forecasts

Dataset Analyst's Findings:
{context}

Conversation History:
{history}

Question: {input}

You don't need to write anything that the csv_agent has already provided, most importantly the data overview like the head stuffs.
Give a detailed economic analysis in the `output` field based on the user input and the context.
Give a clear, head-shot of 50-100 words summary of the output in the `insights` field that should give them a clear follow-up for the context of that output.
Provide a detailed economic analysis. Your analysis MUST be formatted as a JSON object strictly following the provided schema.

Analysis:""" # The model will output the JSON here
    )
    
    # Extract context from messages
    context = "No data context yet."
    history = ""
    
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            content = msg.content
            # Check for data/analysis content
            if "Dataset Analyst" in content or any(keyword in content for keyword in ["imports", "exports", "PKR", "Rupees", "quantity"]):
                context = content
        
        # Build history
        msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Truncate long messages for history
        history += f"\n{msg_type}: {msg.content[:200]}"
    
    formatted_prompt = prompt_template.format(
        input=state["input"],
        context=context,
        history=history[-1000:]  # Keep last 1000 chars
    )
    
    try:
        # 2. INVOKE THE STRUCTURED LLM
        # This returns an instance of the Pydantic model (EconomicsResponse)
        pydantic_response = structured_llm.invoke(formatted_prompt)
        
        # Convert the Pydantic model to a JSON string for storage in the message history
        # You could also store the Pydantic object directly if your state supports it
        json_content = pydantic_response.model_dump_json(indent=2)
        state["messages"].append(AIMessage(content=json_content))
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error during structured output generation: {str(e)}"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state

def create_economics_graph():
    """Create LangGraph for economics agent."""
    graph = StateGraph(EconomicsAgentState)
    graph.add_node("economics_agent", economics_agent_node)
    graph.add_edge("economics_agent", END)
    graph.set_entry_point("economics_agent")
    return graph.compile()

def get_economics_agent():
    """Get compiled economics agent."""
    return create_economics_graph()

def invoke_economics_agent(
    query: str, 
    conversation_history: List[Dict] = None, 
    dataset_context: str = None
):
    """
    Invoke economics agent.
    
    Args:
        query: User question
        conversation_history: Previous messages
        dataset_context: Data from CSV analyst
    
    Returns:
        Economics analysis response
    """
    print("IGI Economics Agent")
    agent = get_economics_agent()
    
    messages = []
    
    # Add conversation history
    if conversation_history:
        for msg in conversation_history:
            if msg.get('role') == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg.get('role') == 'assistant':
                messages.append(AIMessage(content=msg['content']))
    
    # Add dataset context
    if dataset_context:
        messages.append(AIMessage(content=f"Dataset Analysis: {dataset_context}"))
    
    # Invoke agent
    result = agent.invoke({
        "messages": messages,
        "tools": [],
        "input": query,
        "agent_outcome": {},
        "agent_scratchpad": ""
    })
    # Extract response
    if result["messages"]:
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            response = last_message.content
    
        response_dict = json.loads(response)
        return response_dict
    else :
        return "I couldn't generate an analysis. Please try again."

if __name__ == "__main__":
    # Test
    import json
    # 1. The result is a JSON-formatted string
    response_json_string = invoke_economics_agent(
        query="What are the economic implications of Food Group imports?",
        dataset_context="Food Group imports: 50M PKR in June 2025, up 12% MoM"
    )
    
    print("\n--- Raw JSON String Response ---")
    print(response_json_string)
