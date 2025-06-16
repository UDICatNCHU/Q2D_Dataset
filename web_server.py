from pathlib import Path
from flask import Flask, request, jsonify, render_template

from bm25_retrieval import BM25Retriever, load_index, load_corpus

app = Flask(__name__)

# Preload index and corpus for fraud dataset
_INDEX_PATH = Path(__file__).with_name("fraud_index.json")
_CORPUS_DIR = Path("data") / "fraud"
_INDEX = load_index(_INDEX_PATH)
_BM25 = BM25Retriever(_INDEX)
_DOCS = {doc["id"]: doc["text"] for doc in load_corpus(str(_CORPUS_DIR))}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json(force=True)
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Missing query'}), 400
    top_k = int(data.get('top_k', 5))
    results = _BM25.query(query, top_k)
    formatted = [
        {"doc_id": doc_id, "score": score, "text": _DOCS.get(doc_id, "")}
        for score, doc_id in results
    ]
    return jsonify({'results': formatted})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
