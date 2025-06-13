import json
from pathlib import Path
from typing import List, Dict
import argparse

from bm25_retrieval import BM25Retriever, load_index
from score import load_qrels, compute_scores


def load_queries(path: str) -> List[Dict[str, object]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BM25 on the fraud dataset")
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="number of documents to retrieve for each query",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path('data') / 'fraud'
    index_file = 'fraud_index.json'

    queries_path = data_dir / 'format' / 'queries.json'
    qrels_path = data_dir / 'format' / 'qrels.json'

    # load index
    index = load_index(index_file)
    bm25 = BM25Retriever(index)
    queries = load_queries(str(queries_path))

    top_k = args.top_k
    preds = []
    for q in queries:
        results = bm25.query(q["text"], top_k=top_k)
        doc_ids = [doc_id for score, doc_id in results]
        preds.append({"qid": q["id"], "docids": doc_ids})

    # map for scoring
    preds_map = {p["qid"]: p["docids"] for p in preds}
    qrels_map = load_qrels(str(qrels_path))

    accuracy, mrr = compute_scores(qrels_map, preds_map)
    print(f"Evaluated {len(queries)} queries")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MRR: {mrr:.4f}")

    # print top-3 results with ground truth for each query
    for q in queries:
        qid = q["id"]
        top_docs = preds_map.get(qid, [])[:3]
        gt = qrels_map.get(qid)
        print(f"Q{qid}: top-3 {top_docs} | truth {gt}")


if __name__ == '__main__':
    main()
