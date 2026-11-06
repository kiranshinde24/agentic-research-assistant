import arxiv
import chromadb
from sentence_transformers import SentenceTransformer

QUERY = 'all:"retrieval augmented generation" OR all:"LLM agents"'
MAX_PAPERS = 300

client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
search = arxiv.Search(query=QUERY, max_results=MAX_PAPERS,
                      sort_by=arxiv.SortCriterion.Relevance)
papers = list(client.results(search))
print(f"Fetched {len(papers)} papers")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="data/chroma")
col = db.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})

ids, docs, metas = [], [], []
for p in papers:
    ids.append(p.get_short_id())
    docs.append(f"{p.title}. {p.summary.replace(chr(10), ' ')}")
    metas.append({"title": p.title, "url": p.entry_id,
                  "published": str(p.published.date())})

embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=True).tolist()
col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
print("Stored", col.count(), "documents")