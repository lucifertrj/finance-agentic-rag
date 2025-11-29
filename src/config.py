import os
from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

os.environ["WANDB_API_KEY"] = WANDB_API_KEY or ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""

EMBED_MODEL_NAME = "text-embedding-3-small"
COLLECTION_NAME = "financialv2"
WEAVE_PROJECT = "financialv1"
LLM_MODEL = "gpt-4o-mini"
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
TAG_GENERATOR_MODEL = "gemini-2.5-flash-lite"
DATA_PATH = "data"

