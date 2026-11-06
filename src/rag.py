import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from retrieval import retrieve

load_dotenv()
llm = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
from groq import Groq
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"]) if os.getenv("GROQ_API_KEY") else None
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Tries models in order; edit this list using names shown in Google AI Studio
MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash"
).split(",")]


def generate(prompt, retries=3):
    last_err = None
    for model in MODELS:
        for attempt in range(retries):
            try:
                return llm.models.generate_content(model=model, contents=prompt).text
            except (errors.ServerError, errors.ClientError) as e:
                last_err = e
                if getattr(e, "code", None) in (429, 500, 503):
                    time.sleep(2 ** attempt)
                    continue
                break  # bad model name etc.: try the next model

    # Reached only after every Gemini model has failed
    if groq_client:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content
    raise RuntimeError(f"All models failed: {last_err}")

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
    text = generate(prompt)
    return text, hits


if __name__ == "__main__":
    q = input("Question: ")
    text, hits = answer(q)
    print("\n" + text)
    print("\nSources:")
    for i, (_, m) in enumerate(hits):
        print(f"[{i+1}] {m['title']} - {m['url']}")