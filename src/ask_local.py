import os
import subprocess
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

load_dotenv()

# This is the vector database
DB_DIR = "./chroma_db"

''' 
This is a ssmall wrapper to match what chroma expects
'''
class LocalEmbeddingFunction:
    # Loads the embedding model into memory: "all-MiniLM-L6-v2" by default
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)


    # Takes in a list of strings
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    # Takes list of strings
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]

def build_retriever():
    # Creates an instance of the embedding wrapper
    emb = LocalEmbeddingFunction()

    # Opens Chroma database and using the embedding wrapper
    vectordb = Chroma(
        embedding_function=emb,
        # This is where the database lives
        persist_directory=DB_DIR,
    )
    # Turns the vector store into a retriver object
    return vectordb.as_retriever(search_kwargs={"k": 10})


# Call Ollama 3
def call_ollama_llama3(prompt: str) -> str:
    # Uses the `ollama` CLI to run llama3 and stream JSON lines[web:116][web:117][web:119][web:126]
    proc = subprocess.Popen(
        ["ollama", "run", "llama3"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=prompt)
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama error: {stderr}")

    return stdout.strip()

def answer_with_local_llm(question: str, retriever) -> str:
    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find any relevant notes for that question."

    context_blocks = []
    for d in docs:
        meta = d.metadata or {}
        title = meta.get("source") or meta.get("file_path") or "note"
        context_blocks.append(f"[{title}]\n{d.page_content}")

    context_text = "\n\n---\n\n".join(context_blocks)

    recommendation_keywords = ("recommend", "suggest", "what should i read", "what book", "next book", "read next", "books based on", "good books")
    is_recommendation = any(kw in question.lower() for kw in recommendation_keywords)

    if is_recommendation:
        # Extract themes from the notes to use as the basis for recommendations
        # rather than passing raw chunks that distract the model
        themes_prompt = f"""Read these notes and list the key intellectual themes and topics in 3-5 bullet points. Be concise.

Notes:
{context_text}

Themes:"""
        themes = call_ollama_llama3(themes_prompt.strip())

        prompt = f"""You are a book recommendation engine. Elliott is a reader interested in philosophy and ideas.

Based on these themes from his notes:
{themes}

Recommend exactly 4 books he has NOT already read. These must be real, published books.
Do NOT recommend Plato, Marcus Aurelius, Socrates, or anything already implied by the themes above.

For each book, use this exact format:
1. [Title] by [Author] — [2 sentences on why it fits his interests]

Begin immediately with "1." — no preamble, no commentary."""
    else:
        prompt = f"""You are a helpful assistant with access to Elliott's personal notes.

STRICT RULES:
- Only use information explicitly present in the notes below. Do not add, infer, or invent anything.
- If the notes don't contain enough to answer, say "I couldn't find enough in your notes on that" and stop.
- Do not ramble. Answer the question directly and concisely.
- Do not mention RAG, AI systems, or how this tool works unless the question is specifically about that.

Notes:
{context_text}

Question: {question}

Answer:"""

    return call_ollama_llama3(prompt.strip())

def main():
    retriever = build_retriever()
    print("Local RAG with llama3. Type 'quit' to exit.")
    while True:
        # We take in the question from user input
        q = input("\nQuestion: ")
        if q.strip().lower() == "quit":
            break
        ans = answer_with_local_llm(q, retriever)
        print("\nAnswer:\n")
        print(ans)

if __name__ == "__main__":
    main()
