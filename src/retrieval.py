import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
col = chromadb.PersistentClient(path="data/chroma").get_collection("papers")
PREFIX = "Represent this sentence for searching relevant passages: "

# Load all docs once for BM25
_all = col.get(include=["documents", "metadatas"])
IDS, DOCS, METAS = _all["ids"], _all["documents"], _all["metadatas"]
IDX = {id_: i for i, id_ in enumerate(IDS)}


def tok(t):
    return re.findall(r"\w+", t.lower())


bm25 = BM25Okapi([tok(d) for d in DOCS])


def dense_search(q, k=20):
    emb = embedder.encode(PREFIX + q, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[emb], n_results=k)
    return res["ids"][0]


def bm25_search(q, k=20):
    scores = bm25.get_scores(tok(q))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [IDS[i] for i in top]


def rrf(rank_lists, c=60):
    """Reciprocal Rank Fusion: merge several ranked lists."""
    score = {}
    for lst in rank_lists:
        for r, id_ in enumerate(lst):
            score[id_] = score.get(id_, 0) + 1 / (c + r + 1)
    return sorted(score, key=score.get, reverse=True)


def retrieve(q, k=5, use_hybrid=True, use_rerank=True):
    if use_hybrid:
        cand = rrf([dense_search(q), bm25_search(q)])[:20]
    else:
        cand = dense_search(q, k=20)
    if use_rerank:
        scores = reranker.predict([(q, DOCS[IDX[c]]) for c in cand])
        cand = [c for _, c in sorted(zip(scores, cand), key=lambda x: x[0], reverse=True)]
    return [(DOCS[IDX[c]], METAS[IDX[c]]) for c in cand[:k]]