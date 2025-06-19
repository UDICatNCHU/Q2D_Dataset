"""Autonomous keyword tuning agent using the MCP server tools.

This script demonstrates a very small loop where an "agent" observes the
retrieval results of the first three test queries and repeatedly tries to
improve them using the ``search`` and ``expand_search`` tools exposed by the
MCP server.  Accuracy and MRR are measured after every attempt and the query is
updated whenever an expansion yields a better score.


import json
import sys

from pathlib import Path
from typing import Dict, List, Tuple

from mcp_client import MCPClient
from score import load_qrels, compute_scores

DATA_DIR = Path("data") / "fraud" / "format"
QUERIES_PATH = DATA_DIR / "queries.json"
QRELS_PATH = DATA_DIR / "qrels.json"

TOP_K = 5
NUM_QUERIES = 3


def load_queries() -> List[Dict[str, object]]:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_search(client: MCPClient, query: str) -> List[int]:
    resp = client.call_tool("search", {"query": query, "top_k": TOP_K})
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return [item["doc_id"] for item in resp["result"]]


def run_expand_search(client: MCPClient, query: str) -> Tuple[str, List[int]]:
    resp = client.call_tool("expand_search", {"query": query, "top_k": TOP_K})
    if "error" in resp:
        # Gemini might be unavailable; fall back to original query
        print(f"Expansion failed: {resp['error']}", file=sys.stderr)

        expanded_query = query
        docs = run_search(client, query)
    else:
        result = resp["result"]
        expanded_query = result.get("expanded_query", query)
        docs = [item["doc_id"] for item in result.get("results", [])]
    return expanded_query, docs


def evaluate_single(qid: int, rel_doc: int, docs: List[int]) -> Tuple[float, float]:
    """Return accuracy and MRR for one query."""
    qrels = {qid: rel_doc}
    preds = {qid: docs}
    return compute_scores(qrels, preds)


def refine_query(
    client: MCPClient, qid: int, query: str, rel_doc: int, max_iter: int = 3
) -> Tuple[str, List[int]]:
    """Iteratively expand the query if it improves MRR."""

    best_query = query
    docs = run_search(client, query)
    best_acc, best_mrr = evaluate_single(qid, rel_doc, docs)

    for _ in range(max_iter):
        expanded, new_docs = run_expand_search(client, best_query)
        acc, mrr = evaluate_single(qid, rel_doc, new_docs)
        if mrr > best_mrr or (mrr == best_mrr and acc > best_acc):
            best_query = expanded
            docs = new_docs
            best_acc, best_mrr = acc, mrr
        else:
            break

    return best_query, docs


def main() -> None:
    queries = load_queries()
    qrels = load_qrels(str(QRELS_PATH))

    preds_before: Dict[int, List[int]] = {}
    preds_after: Dict[int, List[int]] = {}
    expansions: Dict[int, Tuple[str, str]] = {}

    with MCPClient("mcp_server.py") as client:
        for q in queries[:NUM_QUERIES]:
            qid = q["id"]
            text = q["text"]
            rel_doc = qrels.get(qid)

            preds_before[qid] = run_search(client, text)

            if rel_doc is None:
                preds_after[qid] = preds_before[qid]
                expansions[qid] = (text, text)
                continue

            tuned_query, docs = refine_query(client, qid, text, rel_doc)
            preds_after[qid] = docs
            expansions[qid] = (text, tuned_query)


    # Only evaluate the queries we processed
    subset_qrels = {qid: qrels[qid] for qid in preds_before.keys() if qid in qrels}

    acc_before, mrr_before = compute_scores(subset_qrels, preds_before)
    acc_after, mrr_after = compute_scores(subset_qrels, preds_after)

    print("Original metrics:")
    print(f"  Accuracy: {acc_before:.4f}, MRR: {mrr_before:.4f}")
    print("Expanded metrics:")
    print(f"  Accuracy: {acc_after:.4f}, MRR: {mrr_after:.4f}")
    print()
    for qid, (orig, expanded) in expansions.items():
        print(f"Query {qid}")
        print(f"  Original : {orig}")
        print(f"  Expanded : {expanded}")
        print("-" * 40)


if __name__ == "__main__":
    main()
