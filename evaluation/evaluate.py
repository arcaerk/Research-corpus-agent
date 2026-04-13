import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agent import app, retriever, AgentState

EVALUATION_DATASET = [
    {
        "query": "What does the paper 'Attention Is All You Need' say about training hardware?",
        "expected_title": "Attention Is All You Need",
        "type": "Direct Retrieval"
    },
    {
        "query": "Compare the architecture of Linformer and Reformer.",
        "expected_title": "Linformer: Self-Attention with Linear Complexity",
        "type": "Complex Comparison"
    },
    {
        "query": "What are the core mechanisms of the BERT model?",
        "expected_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "type": "Direct Retrieval"
    },
    {
        "query": "What is the best recipe for baking a chocolate cake?",
        "expected_title": "NONE",
        "type": "Out of Domain (Should Reject)"
    }
]

def run_evaluation():
    print("="*60)
    print("🚀 STARTING AUTOMATED AGENT EVALUATION")
    print("="*60)
    
    total_queries = len(EVALUATION_DATASET)
    successful_retrievals = 0
    passed_evaluations = 0
    
    report_lines = ["# RAG Agent Evaluation Report\n"]
    
    for i, test in enumerate(EVALUATION_DATASET, 1):
        query = test["query"]
        expected = test["expected_title"]
        q_type = test["type"]
        
        print(f"\n[{i}/{total_queries}] Testing Query: '{query}'")
        print(f"Type: {q_type}")
        
        print(" -> Testing Retriever (Recall@3)...")
        test_plan = [expected] if expected != "NONE" else ["chocolate cake recipe"]
        retrieved_docs = retriever.retrieve_and_rerank(test_plan[0], top_k=3)
        
        retrieved_titles = [doc.metadata.get('title', '') for doc in retrieved_docs]
        
        recall_success = expected in retrieved_titles
        if recall_success:
            successful_retrievals += 1
            print("    ✅ Retrieval SUCCESS: Expected paper found in Top 3.")
        elif expected == "NONE":
            print("    ℹ️ N/A (Out of Domain test)")
        else:
            print(f"    ❌ Retrieval FAILED: Expected '{expected}', got: {retrieved_titles}")

        print(" -> Testing Agentic Workflow...")
        initial_state = {
            "query": query, 
            "plan": [], 
            "context": [], 
            "final_answer": "",
            "evaluation": "",
            "loop_count": 0
        }
        
        start_time = time.time()
        final_state = app.invoke(initial_state)
        latency = round(time.time() - start_time, 2)
        
        final_answer = final_state['final_answer']
        critic_decision = final_state['evaluation']
        
        if critic_decision == "pass" or (q_type == "Out of Domain (Should Reject)" and critic_decision == "reject"):
            passed_evaluations += 1
            print(f"    ✅ Agent Execution SUCCESS (Decision: {critic_decision.upper()}, Time: {latency}s)")
        else:
            print(f"    ❌ Agent Execution FAILED (Decision: {critic_decision.upper()})")

        report_lines.append(f"## Test {i}: {q_type}")
        report_lines.append(f"**Query:** {query}")
        report_lines.append(f"**Expected Paper:** {expected}")
        report_lines.append(f"**Recall@3 Match:** {'Yes' if recall_success else 'No'}")
        report_lines.append(f"**Critic Decision:** {critic_decision.upper()}")
        report_lines.append(f"**Latency:** {latency}s")
        report_lines.append(f"**Final Answer:**\n> {final_answer}\n")
        report_lines.append("---\n")

    retrieval_total = total_queries - 1 
    recall_at_3 = (successful_retrievals / retrieval_total) * 100
    agent_success_rate = (passed_evaluations / total_queries) * 100
    
    print("\n" + "="*60)
    print("📊 EVALUATION METRICS SUMMARY")
    print("="*60)
    print(f"Recall@3:           {recall_at_3:.2f}%")
    print(f"Agent Success Rate: {agent_success_rate:.2f}%")
    
    report_lines.insert(1, f"**Overall Recall@3:** {recall_at_3:.2f}%\n**Overall Agent Success Rate:** {agent_success_rate:.2f}%\n\n---\n")
    
    report_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n📝 Detailed Markdown report saved to: {report_path}")

if __name__ == "__main__":
    run_evaluation()