import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# dummy values so consts imports without a real .env
os.environ.setdefault("LLMOD_AI_API_KEY", "test-key")
os.environ.setdefault("LLMOD_AI_URL", "https://example.test/v1")
os.environ.setdefault("PINECONE_API_KEY", "test-pc-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")
