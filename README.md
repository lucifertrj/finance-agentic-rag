# Finance Agentic RAG

An optimized Financial Agentic RAG system for processing SEC filings (10-K, 10-Q, 8-K) and shareholder letters. Built to answer financial queries, summarize documents, and fetch real-time news. The Agent architecture routes queries intelligently between the knowledge base, summarizer, and web search.

<img width="1024" height="960" alt="b19d3143" src="https://github.com/user-attachments/assets/58edddd2-4c6d-4e0d-9f05-6b9ab0d162ce" />

## Core Features

### 1. Qdrant Binary Quantization + RRF Fusion + Hybrid Search

We use Qdrant with binary quantization for efficient storage. Retrieval combines dense (OpenAI embeddings) and sparse (BM25) vectors using Reciprocal Rank Fusion for better recall than either alone.

### 2. Custom Metadata for Summary and Filtering

Each chunk gets enriched with:
- Auto-generated summaries via DistilBART
- Financial tags extracted using Gemini (revenue, subscribers, operating_margin, etc.)
- Document type classification (10-K Filing, 8-K Filing, etc.)

This enables filtered retrieval - ask for "10-K summary" and it pulls only 10-K chunks.

### 3. LangGraph Agent Orchestration
Router agent decides between knowledge_base, summarizer, or web_search tools.

### 4. Weave Observability
Full tracing of agent decisions and LLM calls via Weights & Biases.

### 5. Tavily Web Search
Fallback for queries outside the knowledge base.

<img width="496" height="432" alt="8d85dda5" src="https://github.com/user-attachments/assets/0cb640c8-294b-46cb-84d3-30d78dccbe50" />

🔗 This project belongs to Weights & Biases. Collaborated to write the article: [wandb.ai> Building-a-financial-agentic-RAG-pipeline](https://wandb.ai/ai-team-articles/finance-agentic-rag/reports/Building-a-financial-agentic-RAG-pipeline-Part-1---VmlldzoxNTAwNDkzMQ)

## Snapshots- Weave

<img width="2048" height="1072" alt="fc06c2a3" src="https://github.com/user-attachments/assets/7fbfe587-7f5e-4d9f-91ce-c26cf56c3111" />

<img width="2048" height="1311" alt="f1a2afd7" src="https://github.com/user-attachments/assets/d91f81ad-3485-4ec2-881d-f1fc9d6b0f68" />

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env 
```

## Usage

```python
from src.ingest import run_ingestion
from src.main import run_query

run_ingestion("data") # just once

result = run_query("What were the key drivers of subscriber growth in Netflix's 10-K?")
print(result['response'])
```

## Project Structure

```
src/
├── config.py      # Environment variables
├── client.py      # Qdrant, OpenAI, Gemini clients
├── utils.py       # Summarization, tagging utilities
├── ingest.py      # Document loading and indexing
├── retrieval.py   # Hybrid search
├── agents.py      # Agent nodes and routing logic
├── graph.py       # LangGraph workflow
└── main.py        # Entry point
```
