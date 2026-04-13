import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from agent import app as agent_app

app = FastAPI(title="M.Tech RAG Agent API")

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"Received query: {request.query}")
    
    initial_state = {
        "query": request.query, 
        "plan": [], 
        "context": [],
        "sources": [], 
        "final_answer": "",
        "evaluation": "",
        "loop_count": 0
    }
    
    final_state = agent_app.invoke(initial_state)
    
    return {
        "answer": final_state['final_answer'],
        "critic_decision": final_state['evaluation'],
        "search_plan": final_state['plan'],
        "sources": final_state.get('sources', [])
    }

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")