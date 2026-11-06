import os
import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
llm = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient(path="data/chroma").get_collection("papers")
PREFIX = "Represent this sentence for searching relevant passages: "

def retrieve(question, k=5):
    emb = embedder.encode(PREFIX + question, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[emb], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))

def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{i+1}] {meta['title']} ({meta['url']})\n{doc}"
        for i, (doc, meta) in enumerate(hits)
    )
    prompt = (
        "Answer using ONLY the sources below. Cite sources like [1], [2]. "
        "If the sources don't contain the answer, say you don't know.\n\n"
        f"SOURCES:\n{context}\n\nQUESTION: {question}"
    )
    resp = llm.models.generate_content(model=MODEL, contents=prompt)
    return resp.text, hits

if __name__ == "__main__":
    q = input("Question: ")
    text, hits = answer(q)
    print("\n" + text)
    print("\nSources:")
    for i, (_, m) in enumerate(hits):
        print(f"[{i+1}] {m['title']} - {m['url']}")