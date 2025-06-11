
"""BM25 retrieval using prebuilt index.

Usage:
    python bm25_retrieval.py INDEX_FILE QUERY [TOP_K]

INDEX_FILE should be built with build_bm25_index.py.
"""
import json
from collections import Counter
import sys


class BM25Retriever:
    def __init__(self, index, k1=1.5, b=0.75):
        self.doc_ids = index["doc_ids"]
        self.docs = index["docs"]
        # ensure idf values are floats
        self.idf = {k: float(v) for k, v in index["idf"].items()}
        self.avgdl = index["avgdl"]

        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text):
        # simple character based tokenizer, remove spaces and line breaks
        return [ch for ch in text if not ch.isspace()]

    def score(self, query_tokens, index):
        doc = self.docs[index]
        freqs = Counter(doc)
        score = 0.0
        for w in query_tokens:
            if w not in self.idf:
                continue
            df = freqs.get(w, 0)
            denom = df + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            score += self.idf[w] * df * (self.k1 + 1) / (denom + 1e-8)
        return score

    def query(self, text, top_k=5):
        q_tokens = self._tokenize(text)
        scores = []
        for idx in range(self.N):
            scores.append((self.score(q_tokens, idx), self.doc_ids[idx]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

def load_corpus(data_dir):
    path = Path(data_dir) / "format" / "corpus.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
      
    index_file, query = sys.argv[1], sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    index = load_index(index_file)
    bm25 = BM25Retriever(index)
    results = bm25.query(query, top_k)
    for score, doc_id in results:
        print(f"doc_id: {doc_id}\tscore: {score:.4f}")

if __name__ == "__main__":
    main()
