import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

from retriever import ArXivHybridRetriever

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN not found in .env file!")

class AgentState(TypedDict):
    query: str
    plan: List[str]
    context: List[str]
    sources: List[str]
    final_answer: str
    evaluation: str
    loop_count: int


print("Connecting to Hugging Face Inference API...")

# raw_llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     max_new_tokens=512,
#     temperature=0.1,
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# )


# llm = ChatHuggingFace(llm=raw_llm)
from langchain_openai import ChatOpenAI

def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-120b", 
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        
        temperature=0.1, 
        max_tokens=2000
    )

llm = get_groq_llm()
retriever = ArXivHybridRetriever()

def planner_node(state: AgentState):
    print("\n--- NODE: PLANNER ---")
    query = state['query']
    
    messages = [
        SystemMessage(content="""You are a Research Planner building search queries for an ArXiv database. 
        Rule 1: If the user asks about a specific paper, your FIRST query MUST be EXACTLY the title of that paper and NOTHING ELSE.
        Rule 2: Your SECOND query can combine the paper title with specific abstract terminology (e.g., 'GPUs', 'training time').
        Rule 3: Break the question into exactly 2 distinct search queries. DO NOT use boolean operators, quotation marks, or parentheses.
        
        Example Question: What does the BERT paper say about training hardware?
        Example Output:
        1. BERT
        2. BERT GPUs TPUs training time
        """),
        HumanMessage(content=f"Question: {query}")
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    state['plan'] = [line.strip() for line in content.split('\n') if line.strip() and line[0].isdigit()]
    return state

def researcher_node(state: AgentState):
    print(f"--- NODE: RESEARCHER (Executing {len(state['plan'])} tasks) ---")
    all_docs = []
    unique_sources = set()
    
    for sub_query in state['plan']:
        print(f"Searching: {sub_query}")
        docs = retriever.retrieve_and_rerank(sub_query, top_k=3)
        
        for d in docs:
            title = d.metadata.get('title', 'Unknown Title')
            all_docs.append(f"From '{title}': {d.page_content}")
            unique_sources.add(title) 
    
    state['context'] = all_docs
    state['sources'] = list(unique_sources)
    return state

def synthesis_node(state: AgentState):
    print("--- NODE: ANALYST (Synthesizing Answer) ---")
    context_str = "\n\n".join(state['context'])
    query = state['query']
    
    messages = [
        SystemMessage(content="""You are a strict AI Research Analyst. 
        Rule 1: You must ONLY answer the question using the provided context. 
        Rule 2: DO NOT use your pre-trained knowledge or outside information under any circumstances.
        Rule 3: If the provided context does not contain the answer, you must reply EXACTLY with the phrase: INSUFFICIENT_CONTEXT. Do not write anything else."""),
        HumanMessage(content=f"Context: {context_str}\n\nQuestion: {query}")
    ]
    
    response = llm.invoke(messages)
    state['final_answer'] = response.content.strip()
    return state

def critic_node(state: AgentState):
    print("\n--- NODE: CRITIC (Evaluating Output) ---")
    
    loop_count = state.get('loop_count', 0)
    state['loop_count'] = loop_count + 1
    
    query = state['query']
    answer = state['final_answer']
    
    messages = [
        SystemMessage(content="""You are a strict grading critic. Evaluate the provided query and answer.
        PRIORITY 1 (Domain): If the User Query is  NOT related to Machine Learning, AI, or Computer Science, reply strictly with the word: REJECT. (Ignore all other rules).
        PRIORITY 2 (Context): If the Answer is "INSUFFICIENT_CONTEXT", reply strictly with the word: FAIL.
        PRIORITY 3 (Success): If the Answer successfully addresses the Query using academic language, reply strictly with the word: PASS.

        Reply with ONLY ONE WORD (REJECT, FAIL, or PASS)."""),
        HumanMessage(content=f"Query: {query}\n\nAnswer: {answer}")
    ]
    
    response = llm.invoke(messages)
    decision = response.content.strip().upper()
    
    if "REJECT" in decision:
        print(">> Critic Decision: OUT OF DOMAIN. Rejecting query.")
        state['final_answer'] = "Query Rejected: I am a specialized AI Research Agent. I only answer questions related to Machine Learning, Artificial Intelligence, and Computer Science."
        state['evaluation'] = "reject"
        
    elif "FAIL" in decision:
        print(">> Critic Decision: FAIL. The answer was insufficient.")
        state['evaluation'] = "fail"
        
    else:
        print(">> Critic Decision: PASS. The answer is valid.")
        state['evaluation'] = "pass"
        
    return state

def route_critic(state: AgentState):
    """Reads the critic's evaluation and decides the next node."""
    decision = state.get('evaluation', 'pass')
    loop_count = state.get('loop_count', 0)
    
    if decision == "reject":
        return "end"
        
    if decision == "fail" and loop_count >= 2:
        print(">> Router: Max loops reached. Ending early to prevent infinite loop.")
        return "end"
        
    if decision == "fail":
        print(">> Router: Looping back to Planner to try different search strategies...")
        return "loop"
        
    return "end"


workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("critic", critic_node)


workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "synthesis")
workflow.add_edge("synthesis", "critic")


workflow.add_conditional_edges(
    "critic",
    route_critic,
    {
        "loop": "planner",
        "end": END
    }
)

app = workflow.compile()


if __name__ == "__main__":

    user_input = "What does paper 'Attention Is All You Need' say about training hardware?"

    initial_state = {
        "query": user_input, 
        "plan": [], 
        "context": [], 
        "final_answer": "",
        "evaluation": "",
        "loop_count": 0
    }
    
    print(f"\nProcessing Query: '{user_input}'")
    final_output = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("FINAL AGENT RESPONSE:")
    print("="*50)
    print(final_output['final_answer'])