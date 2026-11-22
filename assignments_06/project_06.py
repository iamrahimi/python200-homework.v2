from pathlib import Path
import os

from dotenv import load_dotenv

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# --- Step 1: Setup ---

# Load environment variables from .env
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# project_06.py is inside assignments_06/, so move up one directory to the
# repository root, then follow the lesson's document path.
project_root = Path(__file__).resolve().parent.parent
docs_dir = project_root / "assignments_06" / "resources" / "groundwork_docs"

assert docs_dir.is_dir(), f"Document directory not found: {docs_dir}"


# --- Step 2: Load the Documents ---

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"\nLoaded {len(documents)} documents:")
for document in documents:
    print("-", document.metadata.get("file_name", "Unknown file"))


# --- Step 3: Build the Index and Query Engine ---

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


def print_source_node(source, number=1):
    """Print a retrieved source's name, score, and text preview."""
    metadata = source.node.metadata
    file_name = metadata.get("file_name", "Unknown file")
    chunk_text = source.node.get_content()

    print(f"Source {number}: {file_name}")
    print(f"Similarity score: {source.score}")
    print(f"Chunk preview: {chunk_text[:200]}")


# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)

    print(f"\nQuestion: {question}")
    print("Answer:", response.response)

    if response.source_nodes:
        print_source_node(response.source_nodes[0])
    else:
        print("No source nodes were retrieved.")

# After running, add your observations here:
# Did the answers match the source documents? Did the assistant sound confident?
# Note anything surprising or any answer that should be checked against its source.


# --- Step 5: Find a Failure ---

# This asks for information that is unlikely to be in the shop's documents.
failure_question = "What is the Wi-Fi password for customers?"
failure_response = query_engine.query(failure_question)

print(f"\nFailure test question: {failure_question}")
print("Full response:", failure_response.response)
print("\nAll three retrieved source nodes:")

for number, source in enumerate(failure_response.source_nodes, start=1):
    print_source_node(source, number)

# After running, describe what happened:
# I chose this question because the supplied documents may not include a Wi-Fi
# password. Did the assistant say it could not find the answer, or did it guess?
# Did the retrieved chunks support the answer? What does its tone suggest about
# trusting an answer without checking its sources? Consider adding an instruction
# to abstain when the retrieved context does not support an answer.


# --- Step 6: Reflection ---

# 1. The core LlamaIndex pipeline takes only a few lines: configure the embedding
# model, load documents, build the index, and create the query engine. A
# keyword-based RAG pipeline requires code to clean and compare words, score
# documents, and select matching text. A framework saves implementation work
# by providing common retrieval steps, while still letting us configure them.
#
# 2. One other useful application would be an assistant for a nonprofit that
# answers staff questions from its current program guides, eligibility rules,
# and service documents.
#
# 3. RAG cannot fully prevent a model from misreading retrieved text or making
# an unsupported inference. Retrieval can find the right passage, but the model
# can still produce an inaccurate answer from it.