# M.Tech AI Research Agent: Full-Stack RAG Architecture

This repository contains a full-stack Retrieval-Augmented Generation (RAG) application built for ARXIV research papers. The system operates as an autonomous agent capable of retrieving, reading, and synthesizing complex ArXiv machine learning papers to answer user queries, complete with citations and automated self-correction.

## 🏗️ Architecture Overview

The system is built on a decoupled JS/Python architecture:

1. **Frontend (Vanilla JS/HTML/CSS):** A responsive, decoupled chat interface that dynamically parses agent responses, metadata, and academic citations.
2. **Backend API (FastAPI):** Serves the frontend and provides the `/api/chat` endpoint to interface with the LangGraph state machine.
3. **Agentic Workflow (LangGraph):** A multi-node state machine consisting of:
   * **Planner Node:** Deconstructs complex user queries into multiple, targeted search strategies.
   * **Researcher Node:** Executes searches and compiles context, tracking exact paper titles for citations.
   * **Analyst Node:** Synthesizes the final academic answer strictly grounded in the retrieved context.
   * **Critic Node:** Acts as a self-correction mechanism. It rejects out-of-domain queries (e.g., recipes) and forces the system to loop and re-search if the context is insufficient.
4. **Hybrid Retriever (LangChain):** Combines dense vector search (ChromaDB + `bge-large-en-v1.5`) with sparse keyword search (BM25) and re-ranks the results using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). Features dynamic hardware detection (CUDA/MPS/CPU).

## 🚀 Features

* **Hardware-Agnostic:** Automatically detects and utilizes Apple Silicon (MPS), NVIDIA GPUs (CUDA), or defaults to CPU.
* **100% Agent Success Rate:** Built-in safeguards prevent pre-training data leakage (hallucinations) and ensure strict grounding.
* **Multi-Query Retrieval:** Automatically executes multiple search paths per prompt for robust context gathering.
* **Enterprise Observability:** Fully integrated with LangSmith for token tracking, latency monitoring, and execution trace visualization.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.9+
* API Keys for Groq (LLM Inference) and LangSmith (Observability).

### 2. Environment Setup
Clone the repository and navigate to the project root:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your keys:

```env
# Inference Provider
GROQ_API_KEY=gsk_your_groq_key_here

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_key_here
LANGCHAIN_PROJECT="MTech_RAG_Agent"
```

---

## 💾 Data Ingestion

Before running the agent, you must build the local vector database from the Hugging Face ArXiv dataset.

```bash
# Ensure your virtual environment is active
python src/ingest.py
```
*Note: This will download the dataset, chunk the abstracts, generate embeddings, and save the persistent Chroma database to `data/chroma_db`.*

---

## 🏃‍♂️ Running the Application

### Start the Web Server
Launch the FastAPI backend and frontend interface using Uvicorn:

```bash
uvicorn api.main:app --reload
```

* **Web Interface:** Open `http://localhost:8000` in your browser.
* **API Documentation:** Open `http://localhost:8000/docs` to view the interactive Swagger UI.

---

## 📊 Automated Evaluation

This project includes a deterministic evaluation script to measure `Recall@3` and `Agent Success Rate` against a curated test suite of complex comparisons and out-of-domain trick questions.

To run the test suite and generate the Markdown report:

```bash
python evaluation/evaluate.py
```

The script will output the latency and Critic decisions to the terminal and generate an `evaluation_report.md` file in the root directory.