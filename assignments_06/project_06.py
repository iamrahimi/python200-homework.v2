import os
from pathlib import Path

from dotenv import load_dotenv


# --- Step 1: Set up the API key and document path ---

PROJECT_DIR = Path(__file__).resolve().parent
# Load environment variables from .env
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# Groundwork Coffee's project documents belong in this folder. This path is
# anchored to this script, so it works when run from the repo root or here.
docs_dir = PROJECT_DIR / "resources" / "groundwork_docs"
assert docs_dir.is_dir(), f"Document directory not found: {docs_dir}"

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Add OPENAI_API_KEY to assignments_06/.env before running.")


# Import the LlamaIndex packages after setup so a missing package gives a clear
# install hint while the script starts.
try:
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
    from llama_index.core.prompts import PromptTemplate
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
except ImportError as error:
    raise ImportError(
        "Install the Week 6 packages with: pip install openai pypdf python-dotenv "
        '\"llama-index-core==0.14.10\" llama-index-embeddings-openai '
        "llama-index-llms-openai llama-index-readers-file"
    ) from error


# --- Step 2: Load and inspect the documents ---

documents = SimpleDirectoryReader(str(docs_dir)).load_data()
assert documents, f"No readable documents found in: {docs_dir}"

document_names = sorted(
    {
        document.metadata.get("file_name", "Unknown file")
        for document in documents
    }
)
print(f"\nLoaded {len(documents)} document pages from {len(document_names)} files:")
for document_name in document_names:
    print("-", document_name)


# --- Step 3: Build the embedding index and query engine ---

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Set the LLM explicitly. Otherwise some LlamaIndex versions may fall back to
# gpt-3.5-turbo, which is deprecated and returns a 404.
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

# This prompt asks the model to stay grounded and acknowledge missing information.
qa_prompt = PromptTemplate(
    """You are a helpful assistant for Groundwork Coffee.
Use only the context below to answer. Do not add outside facts.
If the context does not contain the answer, say that you could not find it in the documents.
Keep the answer clear and concise.

Context:
{context_str}

Question: {query_str}
Answer: """
)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_prompt,
)
print("\nIndex built successfully with text-embedding-3-small.")
print("Query engine is ready (similarity_top_k=3).")


def print_source_node(source, number=1):
    """Print a retrieved chunk's source, similarity score, and text preview."""
    metadata = source.node.metadata or {}
    file_name = metadata.get("file_name", metadata.get("file_path", "Unknown file"))
    chunk_text = " ".join(source.node.get_content().split())
    print(f"  Source {number}: {file_name}")
    print(f"    Similarity score: {source.score}")
    print(f"    Chunk preview: {chunk_text[:200]}")


def print_response_and_sources(response, all_sources=False):
    print("Answer:", response.response)
    if not response.source_nodes:
        print("No source nodes were retrieved.")
        return

    sources = response.source_nodes if all_sources else response.source_nodes[:1]
    for number, source in enumerate(sources, start=1):
        print_source_node(source, number)


# --- Step 4: Ask the five required questions ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

print("\n--- Five Groundwork Coffee questions ---")
for question in questions:
    response = query_engine.query(question)
    print(f"\nQuestion: {question}")
    print_response_and_sources(response)

# Reflection after reviewing the printed answers:
# Compare each answer with its source preview. The first source should support
# the answer; if it does not, the response may be plausible but not grounded.
# Note the model's tone and any irrelevant source returned by retrieval.


# --- LlamaIndex Question 2: Compare top_k=1 with top_k=5 ---

comparison_question = "How does the loyalty program work?"
print("\n--- Retrieval comparison: similarity_top_k=1 and 5 ---")
for top_k in (1, 5):
    comparison_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=qa_prompt,
    )
    comparison_response = comparison_engine.query(comparison_question)
    print(f"\nQuestion (similarity_top_k={top_k}): {comparison_question}")
    print("Answer:", comparison_response.response)
    if comparison_response.source_nodes:
        for number, source in enumerate(comparison_response.source_nodes, start=1):
            metadata = source.node.metadata or {}
            file_name = metadata.get("file_name", "Unknown file")
            print(f"  Source {number}: {file_name}; score={source.score}")
    else:
        print("No source nodes were retrieved.")

# More context is not always better. Compare whether top_k=5 adds useful loyalty
# details or introduces unrelated passages that make the response less focused.


# --- LlamaIndex Question 3: Test a question the documents likely cannot answer ---

hard_question = "What is the owner's favorite lunch restaurant, and why?"
hard_response = query_engine.query(hard_question)
print(f"\n--- Unsupported question: {hard_question} ---")
print_response_and_sources(hard_response, all_sources=True)

# The Groundwork documents describe the business, menu, hours, loyalty program,
# and services; they do not identify an owner's favorite lunch restaurant.
# The prompt tells the model to acknowledge missing information. Check whether
# the answer follows that instruction and whether any retrieved chunks are merely
# loosely related. A similarity threshold plus abstention rules could help further.


# --- Step 5: Find another unsupported answer ---

failure_question = "What is the Wi-Fi password for customers?"
failure_response = query_engine.query(failure_question)
print(f"\n--- Failure test: {failure_question} ---")
print_response_and_sources(failure_response, all_sources=True)

# The supplied Groundwork documents may not contain a customer Wi-Fi password.
# Check whether the assistant admits that, whether it guesses, and whether the
# retrieved passages support its answer. A relevance threshold and abstention
# instruction can help reduce unsupported answers.


# --- Step 6: Evaluate grounded and unsupported answers ---

judge_llm = OpenAI(model="gpt-4o-mini", temperature=0)
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


def evaluate_query(question):
    """Run the query engine and both LLM-as-a-judge evaluators."""
    response = query_engine.query(question)
    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=question,
        response=response,
    )
    relevancy_result = relevancy_evaluator.evaluate_response(
        query=question,
        response=response,
    )

    print(f"\nEvaluation question: {question}")
    print("Answer:", response.response)
    print("Faithfulness score:", faithfulness_result.score)
    print("Faithfulness feedback:", faithfulness_result.feedback)
    print("Relevancy score:", relevancy_result.score)
    print("Relevancy feedback:", relevancy_result.feedback)
    return response, faithfulness_result, relevancy_result


grounded_question = "How does the Groundwork Coffee loyalty program work?"
evaluate_query(grounded_question)

unsupported_question = "What is the Wi-Fi password for customers?"
evaluate_query(unsupported_question)

# A faithfulness score of 1.0 means the judge considers the answer supported by
# the retrieved context; 0.0 means the answer is not supported (or is judged
# unfaithful). Relevancy measures whether the answer addresses the question.
# They differ because an answer can be relevant but unsupported, or supported by
# context while failing to answer what the user asked.
# The unsupported Wi-Fi question is expected to receive lower scores if the
# answer guesses or discusses unrelated text. Scores can differ because its
# retrieved context and answer differ from the loyalty question. LLM-as-a-judge
# uses another model to assess qualities that do not have one exact wording;
# exact-match accuracy would unfairly reject correct answers phrased differently.


# --- Step 7: Project reflection ---

# 1. LlamaIndex is valuable because it handles common RAG steps—document loading,
# chunking, embedding, vector search, and response synthesis—without requiring
# me to build each part manually. I can still choose the embedding model and how
# many source chunks the query engine retrieves.
#
# 2. A nonprofit could use this approach to help staff find current answers in
# program guides, eligibility requirements, and service policies. The assistant
# could point staff to the source documents while reducing time spent searching.
#
# 3. RAG can still fail if the right passage is not retrieved, if the model
# misreads it, or if it draws an unsupported conclusion. I would reduce this risk
# by showing source passages, setting a relevance threshold, and telling the
# assistant to say when its documents do not contain the answer.
