"""Build BM25 index for dataset.

Usage:
    python build_bm25_index.py DATA_DIR OUTPUT_INDEX

DATA_DIR should contain format/corpus.json.
"""
import json
import math
from collections import Counter
import sys
from pathlib import Path


def tokenize(text):
    """Very simple character tokenizer."""
    return [ch for ch in text if not ch.isspace()]


def load_corpus(data_dir: str):
    path = Path(data_dir) / "format" / "corpus.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(corpus):
    doc_ids = [doc["id"] for doc in corpus]
    docs = [tokenize(doc["text"]) for doc in corpus]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N
    df = Counter()
    for doc in docs:
        for w in set(doc):
            df[w] += 1
    idf = {w: math.log(1 + (N - df_w + 0.5) / (df_w + 0.5)) for w, df_w in df.items()}
    return {
        "doc_ids": doc_ids,
        "docs": docs,
        "idf": idf,
        "avgdl": avgdl,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    data_dir, out_file = sys.argv[1], sys.argv[2]
    corpus = load_corpus(data_dir)
    index = build_index(corpus)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Index saved to {out_file}")


if __name__ == "__main__":
    main()
