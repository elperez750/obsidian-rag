import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH")
DB_DIR = "./chroma_db"


class LocalEmbeddingFunction:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]

def main():
    if not VAULT_PATH:
        raise ValueError("Set VAULT_PATH in your .env")



    # Load notes from Obsidian vault
    print(f"Loading notes from {VAULT_PATH} ...")
    loader = ObsidianLoader(VAULT_PATH, collect_metadata=True)
    docs = loader.load()
    print(f"Loaded {len(docs)} notes.")

    # Split notes into chunks
    splitter = RecursiveCharacterTextSplitter(
        # Each chunk is 500 characters
        chunk_size=500,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

      
    # Inspect the first chunk to see what a "piece" of your notes looks like
    sample = chunks[0]
    print("\n=== SAMPLE CHUNK ===")
    print("Metadata:", sample.metadata)
    print("\nContent:\n", sample.page_content[:500], "...")
    print("====================\n")

    print(f"Created {len(chunks)} chunks.")

    embedding_fn = LocalEmbeddingFunction()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=DB_DIR,
    )
    vectordb.persist()
    print("Index built and saved to", DB_DIR)

if __name__ == "__main__":
    main()
