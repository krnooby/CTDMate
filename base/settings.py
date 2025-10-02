import os
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Llama(플래너)
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/models/Llama-3.2-1B-Instruct.Q4_K_M.gguf")
LLAMA_CTX        = int(os.getenv("LLAMA_CTX", "4096"))
LLAMA_THREADS    = int(os.getenv("LLAMA_THREADS", str(os.cpu_count() or 8)))
LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "800"))

# Upstage(Solar/Parser)
UPSTAGE_API_KEY   = os.getenv("UPSTAGE_API_KEY", "")
UPSTAGE_MODEL     = os.getenv("UPSTAGE_GEN_MODEL", "solar-pro2")
UPSTAGE_SPLIT     = os.getenv("UPSTAGE_SPLIT", "page")
UPSTAGE_OUTFMT    = os.getenv("UPSTAGE_OUTPUT_FORMAT", "pdf")

# Qdrant + E5
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qdrant_storage")
E5_MODEL_NAME     = os.getenv("E5_MODEL_NAME", "intfloat/multilingual-e5-large-instruct")
E5_QUERY_PREFIX   = os.getenv("E5_QUERY_PREFIX", "query: ")
QDRANT_PATH       = os.getenv("QDRANT_PATH", "qdrant_storage")


# Paths
OUTPUT_DIR        = Path(os.getenv("OUTPUT_DIR", "./output"))
JSONL_DIR         = Path(os.getenv("JSONL_DIR", "./output"))
UPLOAD_MAX        = int(os.getenv("MAX_FILES", "5"))

# YAML
CHECKLIST_PATH     = os.getenv("CHECKLIST_PATH", "checklist.yaml")
RAG_FILTERS_PATH   = os.getenv("RAG_FILTERS_PATH", "rag_filters.yaml")
NORMALIZATION_PATH = os.getenv("NORMALIZATION_PATH", "normalize.yaml")

# RAG knobs
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "2"))
RAG_SIM_THRESHOLD  = float(os.getenv("RAG_SIM_THRESHOLD", "0.78"))
