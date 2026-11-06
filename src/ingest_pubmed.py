import time
import xml.etree.ElementTree as ET
import requests
import chromadb
from sentence_transformers import SentenceTransformer

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TERM = ('"type 2 diabetes"[Title/Abstract] AND (management OR treatment) '
        'AND hasabstract AND 2020:2026[dp]')
MAX_PAPERS = 500

r = requests.get(f"{BASE}/esearch.fcgi", params={
    "db": "pubmed", "term": TERM, "retmax": MAX_PAPERS,
    "retmode": "json", "sort": "relevance"}, timeout=30)
pmids = r.json()["esearchresult"]["idlist"]
print("Found", len(pmids), "PMIDs")

def text(el):
    return "".join(el.itertext()).strip() if el is not None else ""

ids, docs, metas = [], [], []
for i in range(0, len(pmids), 100):
    batch = pmids[i:i + 100]
    x = requests.post(f"{BASE}/efetch.fcgi", data={
        "db": "pubmed", "id": ",".join(batch), "retmode": "xml"}, timeout=60)
    root = ET.fromstring(x.content)
    for art in root.findall(".//PubmedArticle"):
        pmid = text(art.find(".//PMID"))
        title = text(art.find(".//ArticleTitle"))
        abstract = " ".join(text(a) for a in art.findall(".//AbstractText"))
        year = text(art.find(".//PubDate/Year")) or "n/a"
        if not abstract:
            continue
        ids.append(pmid)
        docs.append(f"{title}. {abstract}")
        metas.append({"title": title,
                      "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                      "published": year})
    time.sleep(0.5)
print("Parsed", len(docs), "abstracts")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="data/chroma")
try:
    db.delete_collection("papers")   # remove the old arXiv data
except Exception:
    pass
col = db.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})
embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=True).tolist()
col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
print("Stored", col.count(), "documents")