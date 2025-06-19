import json
import os
from pathlib import Path
from typing import List, Dict
import sys
try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

from bm25_retrieval import BM25Retriever, load_index, load_corpus
from score import load_qrels, compute_scores


try:
    # Use the real FastMCP implementation if available so that the MCP
    # CLI can recognize this server object.
    from mcp.server.fastmcp.server import FastMCP as MCPServer
except Exception:  # pragma: no cover - optional dependency
    class MCPServer:
        """Minimal FastMCP-like server for demonstration."""

        def __init__(self, name: str):
            self.name = name
            self._tools: Dict[str, callable] = {}

        def tool(self, name: str | None = None):
            def decorator(func):
                self._tools[name or func.__name__] = func
                return func

            return decorator

        def run(self, transport: str = "stdio"):
            if transport != "stdio":
                raise ValueError("Only stdio transport is supported in this demo")
            print(f"{self.name} server ready", flush=True)
            for line in sys.stdin:
                try:
                    req = json.loads(line)
                    tool = req.get("tool")
                    args = req.get("args", {})
                    if tool not in self._tools:
                        resp = {"error": f"unknown tool: {tool}"}
                    else:
                        result = self._tools[tool](**args)
                        resp = {"result": result}
                except Exception as e:
                    resp = {"error": str(e)}
                print(json.dumps(resp), flush=True)


mcp = MCPServer("q2d_search")

# Preload index and corpus for the fraud dataset
_INDEX_PATH = Path(__file__).with_name("fraud_index.json")
_CORPUS_DIR = Path("data") / "fraud"
_INDEX = load_index(_INDEX_PATH)
_BM25 = BM25Retriever(_INDEX)
_DOCS = {doc["id"]: doc["text"] for doc in load_corpus(str(_CORPUS_DIR))}
_QUERIES_PATH = _CORPUS_DIR / "format" / "queries.json"
_QRELS_PATH = _CORPUS_DIR / "format" / "qrels.json"

# Configure Gemini model for query expansion
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_GEMINI_MODEL = None
if _GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=_GEMINI_API_KEY)
        _GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        _GEMINI_MODEL = None

with open(_QUERIES_PATH, "r", encoding="utf-8") as f:
    _QUERIES = json.load(f)

_QRELS = load_qrels(str(_QRELS_PATH))

@mcp.tool()
def read_fraud_data(offset: int = 0, limit: int | None = None) -> List[Dict[str, object]]:
    """Return a slice of the fraud judgment summary dataset.

    Parameters
    ----------
    offset: int, optional
        Starting index of records to return. Defaults to ``0``.
    limit: int | None, optional
        Maximum number of records to return. ``None`` will return all
        records after ``offset``.
    """
    path = _CORPUS_DIR / "fraud_judgment_summary.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if offset < 0:
        offset = 0
    if limit is None or limit <= 0:
        return data[offset:]
    return data[offset : offset + limit]


@mcp.tool()
def read_fraud_queries(offset: int = 0, limit: int | None = None) -> List[Dict[str, object]]:
    """Return a slice of the fraud queries dataset."""
    if offset < 0:
        offset = 0
    if limit is None or limit <= 0:
        return _QUERIES[offset:]
    return _QUERIES[offset : offset + limit]


@mcp.tool()
def test() -> str:
    """Check if the server is running."""
    return "Q2D search server is running"


@mcp.tool()
def search(query: str, top_k: int = 5) -> List[Dict[str, object]]:
    """Return top_k search results from the fraud dataset."""
    results = _BM25.query(query, top_k)
    return [
        {"doc_id": doc_id, "score": score, "text": _DOCS.get(doc_id, "")}
        for score, doc_id in results
    ]


@mcp.tool()
def expand_search(query: str, top_k: int = 5) -> Dict[str, object]:
    """Expand the query using Gemini then search the fraud dataset."""
    if not _GEMINI_MODEL:
        raise RuntimeError("Gemini model is not configured")

    prompt = (
        "請擴充以下查詢，列出可能的同義詞或相關關鍵字，以空格分隔：" f"{query}"
    )
    try:
        resp = _GEMINI_MODEL.generate_content(prompt)
        expanded = resp.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini expansion failed: {e}")

    results = _BM25.query(expanded, top_k)
    formatted = [
        {"doc_id": doc_id, "score": score, "text": _DOCS.get(doc_id, "")}
        for score, doc_id in results
    ]
    return {"expanded_query": expanded, "results": formatted}


@mcp.tool()
def evaluate_fraud(top_k: int = 10) -> Dict[str, float]:
    """Run BM25 on fraud queries and return average scores."""
    preds = {}
    for q in _QUERIES:
        res = _BM25.query(q["text"], top_k)
        preds[q["id"]] = [doc_id for score, doc_id in res]
    accuracy, mrr = compute_scores(_QRELS, preds)
    return {"accuracy": accuracy, "mrr": mrr}


if __name__ == "__main__":
    mcp.run(transport="stdio")
