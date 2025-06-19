import json
from pathlib import Path
from bm25_retrieval import BM25Retriever, load_index
from score import compute_scores, load_qrels


def load_queries(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def main():
    # paths
    data_dir = Path('data') / 'fraud' / 'format'
    queries_path = data_dir / 'queries.json'
    qrels_path = data_dir / 'qrels.json'
    index_file = 'fraud_index.json'

    # load resources
    queries = load_queries(str(queries_path))
    qrels_map = load_qrels(str(qrels_path))
    index = load_index(index_file)
    bm25 = BM25Retriever(index)

    # take the first query
    query = queries[0]
    qid = query['id']
    results = bm25.query(query['text'], top_k=10)
    ranked_docs = [doc_id for score, doc_id in results]
    preds = {qid: ranked_docs}

    accuracy, mrr = compute_scores(qrels_map, preds)

    print('Query:', query['text'])
    print('Top results:', results)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'MRR: {mrr:.4f}')


if __name__ == '__main__':
    main()
