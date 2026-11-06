import os
import time
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

load_dotenv()
llm = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Tries models in order; edit this list using names shown in Google AI Studio
MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-2.5-flash-lite,gemini-2.0-flash"
).split(",")]

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient(path="data/chroma").get_collection("papers")
PREFIX = "Represent this sentence for searching relevant passages: "


def generate(prompt, retries=3):
    last_err = None
    for model in MODELS:
        for attempt in range(retries):
            try:
                return llm.models.generate_content(model=model, contents=prompt)
            except (errors.ServerError, errors.ClientError) as e:
                last_err = e
                code = getattr(e, "code", None)
                if code in (429, 500, 503):
                    time.sleep(2 ** attempt)   # wait 1, 2, 4 s then retry
                    continue
                break                           # e.g. 404 bad model name: try next model
    raise RuntimeError(f"All models failed: {last_err}")


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
    resp = generate(prompt)
    return resp.text, hits


if __name__ == "__main__":
    q = input("Question: ")
    text, hits = answer(q)
    print("\n" + text)
    print("\nSources:")
    for i, (_, m) in enumerate(hits):
        print(f"[{i+1}] {m['title']} - {m['url']}")