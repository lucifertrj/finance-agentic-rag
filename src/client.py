import torch
from transformers import pipeline
from google.genai import Client as GeminiClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from tavily import TavilyClient
import weave

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBED_MODEL_NAME,
    LLM_MODEL,
    SUMMARIZER_MODEL,
    WEAVE_PROJECT,
)

summarizer_pipeline = pipeline(
    "summarization",
    model=SUMMARIZER_MODEL,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    batch_size=8,
    max_length=128,
    truncation=True
)

gemini_client = GeminiClient()

dense_model = OpenAIEmbeddings(model=EMBED_MODEL_NAME)

sparse_model = SparseTextEmbedding(model_name="Qdrant/BM25")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

llm = ChatOpenAI(model=LLM_MODEL)

tavily_client = TavilyClient()

weave_observer = weave.init(WEAVE_PROJECT)

