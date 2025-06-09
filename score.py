import json
import sys
from typing import Dict, List


def load_qrels(path: str) -> Dict[int, int]:
    """Load qrels mapping query id to relevant doc id."""
    with open(path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    mapping = {}
    for entry in qrels:
        mapping[int(entry['qid'])] = int(entry['docid'])
    return mapping


def load_preds(path: str) -> Dict[int, List[int]]:
    """Load predictions mapping query id to ranked doc ids list."""
    with open(path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    mapping = {}
    for entry in preds:
        mapping[int(entry['qid'])] = [int(d) for d in entry['docids']]
    return mapping


def compute_scores(qrels: Dict[int, int], preds: Dict[int, List[int]]):
    """Compute accuracy and MRR."""
    total = len(qrels)
    hit = 0
    reciprocal_rank_sum = 0.0
    for qid, rel_doc in qrels.items():
        ranking = preds.get(qid, [])
        if rel_doc in ranking:
            hit += 1
            rank = ranking.index(rel_doc) + 1
            reciprocal_rank_sum += 1.0 / rank
    accuracy = hit / total if total else 0.0
    mrr = reciprocal_rank_sum / total if total else 0.0
    return accuracy, mrr


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py <qrels_path> <preds_path>")
        sys.exit(1)
    qrels_path, preds_path = sys.argv[1], sys.argv[2]
    qrels = load_qrels(qrels_path)
    preds = load_preds(preds_path)
    accuracy, mrr = compute_scores(qrels, preds)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MRR: {mrr:.4f}")
