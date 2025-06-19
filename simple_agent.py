import json
from pathlib import Path
from bm25_retrieval import BM25Retriever, load_index
from score import compute_scores, load_qrels


def load_queries(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_queries(ground_truth, results):
    """評估單一查詢的準確率和MRR"""
    # 確保 ground_truth 是列表格式
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    
    # 準確率：是否找到至少一個正確文件
    accuracy = 1.0 if any(doc_id in ground_truth for _, doc_id in results) else 0.0
    
    # MRR：第一個正確文件的倒數排名
    for rank, (_, doc_id) in enumerate(results, 1):
        if doc_id in ground_truth:
            mrr = 1.0 / rank
            break
    else:
        mrr = 0.0
    
    return accuracy, mrr


def main():
    # paths
    data_dir = Path('data') / 'fraud' / 'format'
    queries_path = data_dir / 'queries.json'
    qrels_path = data_dir / 'qrels.json'
    index_file = 'fraud_index.json'
    
    # 設定要測試的查詢數量
    k = 10  # 可以修改這個數字

    # load resources
    queries = load_queries(str(queries_path))
    qrels_map = load_qrels(str(qrels_path))
    index = load_index(index_file)
    bm25 = BM25Retriever(index)

    total_accuracy = 0.0
    total_mrr = 0.0
    
    print(f"Testing first {k} queries...")
    print("-" * 50)
    
    # 處理前 k 筆查詢
    for i in range(min(k, len(queries))):
        query = queries[i]
        qid = query['id']
        results = bm25.query(query['text'], top_k=10)

        # get ground truth for this query
        ground_truth = qrels_map.get(qid, [])

        # 評估查詢結果
        accuracy, mrr = evaluate_queries(ground_truth, results)
        
        total_accuracy += accuracy
        total_mrr += mrr
        
        print(f'Query {i+1}: {query["text"][:50]}...')
        print(f'Ground Truth IDs: {ground_truth}')
        print(f'Top result: {results[0] if results else "None"}')
        print(f'Accuracy: {accuracy:.4f}, MRR: {mrr:.4f}')
        print("-" * 50)
    
    # 計算平均分數
    actual_k = min(k, len(queries))
    avg_accuracy = total_accuracy / actual_k
    avg_mrr = total_mrr / actual_k
    
    print(f'\nResults for {actual_k} queries:')
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f'Average MRR: {avg_mrr:.4f}')


if __name__ == '__main__':
    main()
