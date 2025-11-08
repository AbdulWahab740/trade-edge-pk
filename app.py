# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
from pydantic import BaseModel, Field

from agents.csv_agent import csv_analyzer
from agents.economics_agent import invoke_economics_agent
from agents.graph_agent import graph_agent
from utils.extract_json import _extract_chart_json_from_text
# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI
app = FastAPI(
    title="Pakistan Trade Analytics API",
    description="Dual-agent system for trade data analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://trade-edge-pk.vercel.app","https://trade-edge-pr18bvp5o-abdulwahab740s-projects.vercel.app",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== REQUEST/RESPONSE MODELS ====================

class MessageHistory(BaseModel):
    role: str
    content: str
    agent: Optional[str] = None

class ChatAnalystRequest(BaseModel):
    query: str
    conversation_history: Optional[List[MessageHistory]] = Field(default_factory=list)
    dataset_context: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    success: bool
    response: str
    data: Optional[Dict[str, Any]] = None
    chart_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SchemaResponse(BaseModel):
    success: bool
    schema: Optional[str] = None
    columns: Optional[List[str]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    agents: Dict[str, str]

# ==================== HEALTH CHECK ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and agents are available."""
    return HealthResponse(
        status="healthy",
        agents={
            "dataset_analyst": "available",
            "economics_expert": "available"
        }
    )

# ==================== DATASET ANALYST ROUTES ====================
def query_dataset_analyst(query, conversation_history):
    """
    Query the Dataset Analyst (CSV Agent).
    Analyzes the Pakistan trade CSV data.
    """
    try:
        # Convert conversation history to the format expected by the agent
        history = [
            {   
                "role": msg.role,
                "content": msg.content,
                "agent": msg.agent
            }
            for msg in conversation_history
        ]
        
        # Invoke the CSV agent
        response_text = csv_analyzer(query, history)
        return response_text
        
    except Exception as e:
        return str(e)

def query_economics_expert(query, conversation_history, dataset_context):
    """Query the Economics Expert."""
    try:  
        # Convert conversation history
        history = []
        for msg in conversation_history:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "agent": msg.agent
            })
        
        context_text = None
        if dataset_context:
            # If it's a dict, extract the response
            if isinstance(dataset_context, dict):
                context_text = dataset_context.get('response', str(dataset_context))
            else:
                context_text = str(dataset_context)
        
        response_text = invoke_economics_agent(
            query=query,
            conversation_history=history,
            dataset_context=context_text
        )
        
        return response_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return str(e)

def query_graph_agent(query):
    """Query the Graph Agent."""
    try:
        result = graph_agent(query)
        # Unwrap agent executor result to raw output string if necessary
        if isinstance(result, dict) and "output" in result:
            return result["output"]
        return result
    except Exception as e:
        return str(e)


@app.post("/analyze")
async def analyze(request: ChatAnalystRequest):
    """Analyze the dataset."""
    try:
        csv_response = query_dataset_analyst(request.query, request.conversation_history)
        economics_response = query_economics_expert(request.query, request.conversation_history, csv_response)
        graph_response = query_graph_agent(request.query)
        
        # Extract chart_json only
        chart_data: Optional[Dict[str, Any]] = None
        

        # Case 1: output is a JSON string (possibly wrapped in code fences)
        if isinstance(graph_response, str):
            txt = graph_response
            chart_data = _extract_chart_json_from_text(txt)
        # Case 2: output is already a dict
        elif isinstance(graph_response, dict):
            # Direct chart_json
            if "chart_json" in graph_response:
                chart_data = graph_response["chart_json"]
            # Sometimes agent returns { output: "...json..." }
            elif "output" in graph_response and isinstance(graph_response["output"], str):
                chart_data = _extract_chart_json_from_text(graph_response["output"])

        return AgentResponse(
            success=True,
            response=csv_response,
            data={"csv_response": csv_response, "economics_response": economics_response},
            chart_data=chart_data
        )
    except Exception as e:
        return AgentResponse(
            success=False,
            response=str(e),
            error=str(e)
        )

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)