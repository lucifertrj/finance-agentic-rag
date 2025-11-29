# Finance Agentic RAG

A production-ready agentic RAG system for processing SEC filings (10-K, 10-Q, 8-K) and shareholder letters. Built to answer financial queries, summarize documents, and fetch real-time news. The multi-agent architecture routes queries intelligently between knowledge base, summarizer, and web search.

## Core Features

### Qdrant Binary Quantization + RRF Fusion + Hybrid Search

We use Qdrant with binary quantization for efficient storage. Retrieval combines dense (OpenAI embeddings) and sparse (BM25) vectors using Reciprocal Rank Fusion for better recall than either alone.

### Custom Metadata for Summary and Filtering

Each chunk gets enriched with:
- Auto-generated summaries via DistilBART
- Financial tags extracted using Gemini (revenue, subscribers, operating_margin, etc.)
- Document type classification (10-K Filing, 8-K Filing, etc.)

This enables filtered retrieval - ask for "10-K summary" and it pulls only 10-K chunks.

### LangGraph Agent Orchestration
Router agent decides between knowledge_base, summarizer, or web_search tools.

### Weave Observability
Full tracing of agent decisions and LLM calls via Weights & Biases.

### Tavily Web Search
Fallback for queries outside the knowledge base.

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
