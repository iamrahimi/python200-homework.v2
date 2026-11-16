# --- Setup ---
from pathlib import Path
import os
import string

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Confirm API key loaded
if api_key:
    print("API key loaded successfully.")
else:
    print("API key not found.")
    
# --- RAG Concepts ---

# Concepts Q1
# Scenario A — RAG:
# RAG fits because the assistant needs to answer from a large policy library that
# changes quarterly. It can retrieve relevant sections from the latest documents
# without retraining the model every time a policy changes.
#
# Scenario B — Fine-tuning:
# Fine-tuning on the startup's 3,000 examples can teach the model a consistent,
# unusual writing style. Prompt instructions can help guide the voice, but the
# examples provide repeated demonstrations of the style.
#
# Scenario C — Prompt engineering:
# For one short report, the analyst can provide the report in the prompt and ask
# focused questions about it. A reusable retrieval system would add unnecessary
# setup for a one-time document.

# Concepts Q2
# A confidently wrong answer can be more harmful because people may act on it
# without checking. For example, a patient might follow an invented medication
# dosage if the answer sounds authoritative. A clear "I'm not sure" signals that
# the user should verify the information. Tone affects trust: confident wording
# can make unsupported information sound more reliable than it is.

# Concepts Q3
# Original steps, in the order given:
# 1. Generate a response from the LLM
# 2. Extract text from source documents
# 3. Receive the user's query
# 4. Retrieve the most relevant chunks
# 5. Convert text chunks into embeddings
# 6. Inject retrieved chunks into the prompt
# 7. Split text into chunks
# 8. Embed the user's query
#
# Correct pipeline order:
# 1. Extract text from source documents — read usable text from the files.
# 2. Split text into chunks — divide long documents into smaller passages.
# 3. Convert text chunks into embeddings — represent each passage as a vector.
# 4. Receive the user's query — accept the question to answer.
# 5. Embed the user's query — represent the question as a vector.
# 6. Retrieve the most relevant chunks — find passages with similar meaning.
# 7. Inject retrieved chunks into the prompt — provide the passages as context.
# 8. Generate a response from the LLM — answer using the question and context.


# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": (
        "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include "
        "croissants and muffins baked fresh daily. Oat milk and almond milk are available."
    ),
    "hours.txt": (
        "We are open Monday through Friday from 7am to 7pm. On weekends we open "
        "at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day."
    ),
    "hiring.txt": (
        "We are currently hiring baristas and shift supervisors. Send your resume "
        "to jobs@groundworkcoffee.com."
    ),
    "loyalty.txt": (
        "Join our loyalty program to earn one point per dollar spent. Redeem 100 "
        "points for a free drink of your choice."
    ),
}


# Keyword Question 1
query = "What are your hours on weekends?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# The selected document is hours.txt. "weekends" appears in that document, so it
# has the strongest keyword overlap for this query.


# Keyword Question 2
query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# The function returns "None found" because the query's remaining words do not
# appear in any document. Keyword matching misses the likely connection between
# caffeine and the coffee menu. Semantic retrieval may find the menu, though the
# menu itself does not say whether a caffeine-free drink is available.


# Keyword Question 3
# Prediction: I predict loyalty.txt because "sign up for rewards" sounds like
# joining a rewards or loyalty program.
query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# The prediction is not correct: the function returns "None found." The document
# says "loyalty program," not "rewards," and this retriever does not understand
# that those phrases can have similar meanings.


# --- Semantic RAG Concepts ---

# Semantic Question 1
# 1. An embedding is a list of numbers that represents the meaning of text.
#    Similar ideas tend to have vectors that point in similar directions.
# 2. The chunk with a score of 0.85 is more relevant than the one with 0.30.
#    The higher score means its meaning is more closely aligned with the query.
# 3. Semantic search compares the meaning represented by the vectors, so it can
#    match related ideas even when the words are different.

# Semantic Question 2
# | Feature                    | Keyword RAG                       | Semantic RAG                         |
# |----------------------------|-----------------------------------|--------------------------------------|
# | What is compared?          | Exact word overlap                | Query and chunk embedding vectors   |
# | What is retrieved?         | Full document                     | Most relevant text chunks            |
# | Can it handle synonyms?    | No                                | Often, because it compares meaning  |
# | Storage format             | Plain text dictionary             | Embeddings in a vector index         |
# | Relevance score            | Number of overlapping keywords    | Vector similarity, often cosine      |


# --- LlamaIndex ---

def print_sources(response):
    """Print each retrieved source's score and a short text preview."""
    for number, source in enumerate(response.source_nodes, start=1):
        print(f"Source {number} similarity score: {source.score}")
        print(f"Source {number} text: {source.node.get_content()[:150]}")


def run_llamaindex_exercises():
    """Build and query the BrightLeaf semantic RAG pipeline."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\nSkipping LlamaIndex exercises: OPENAI_API_KEY is not set.")
        return

    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.evaluation import (
            FaithfulnessEvaluator,
            RelevancyEvaluator,
        )
        from llama_index.core import SimpleDirectoryReader
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
    except ImportError as error:
        print(f"\nCould not import LlamaIndex packages: {error}")
        return

    # Adjust this path if your lesson materials are stored elsewhere.
    pdf_dir = Path(
        os.getenv(
            "BRIGHTLEAF_PDF_DIR",
            "../../06_AI_augmentation/resources/brightleaf_pdfs",
        )
    )
    if not pdf_dir.is_absolute():
        pdf_dir = (Path(__file__).resolve().parent / pdf_dir).resolve()

    if not pdf_dir.exists():
        print(f"\nBrightLeaf PDF folder was not found: {pdf_dir}")
        print("Set BRIGHTLEAF_PDF_DIR or update the path in this script.")
        return

    try:
        documents = SimpleDirectoryReader(str(pdf_dir)).load_data()
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        answer_llm = OpenAI(model="gpt-4o-mini")

        # LlamaIndex Question 1
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            llm=answer_llm,
        )

        questions = [
            "What employee benefits does BrightLeaf offer?",
            "What are BrightLeaf's security policies?",
        ]

        for question in questions:
            print(f"\nQuestion: {question}")
            response = query_engine.query(question)
            print("Answer:", response.response)
            print_sources(response)

        # After running, record whether the chunks fit each question, what tone
        # the model used, and whether any retrieved chunk seemed unexpected.
        # Results depend on the BrightLeaf PDFs and the model's response.

        # LlamaIndex Question 2
        question = "What employee benefits does BrightLeaf offer?"
        for top_k in (1, 5):
            engine = index.as_query_engine(
                similarity_top_k=top_k,
                llm=answer_llm,
            )
            response = engine.query(question)
            print(f"\nQuestion with similarity_top_k={top_k}: {question}")
            print("Answer:", response.response)
            print_sources(response)

        # After running, compare the answers and retrieved scores. More context
        # can add useful details, but it can also add unrelated passages.

        # LlamaIndex Question 3
        difficult_question = "What is BrightLeaf's CEO's favorite food?"
        difficult_engine = index.as_query_engine(
            similarity_top_k=5,
            llm=answer_llm,
        )
        response = difficult_engine.query(difficult_question)
        print(f"\nQuestion: {difficult_question}")
        print("Answer:", response.response)
        print_sources(response)

        # After running, note whether the system admitted that the documents did
        # not answer this. Possible improvements include an abstention rule,
        # retrieval thresholds, and checking whether the context supports claims.

        # LlamaIndex Question 4
        judge_llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
        faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
        relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

        evaluation_questions = [
            "What employee benefits does BrightLeaf offer?",
            "What is BrightLeaf's CEO's favorite food?",
        ]

        for question in evaluation_questions:
            response = query_engine.query(question)
            faithfulness = faithfulness_evaluator.evaluate_response(
                query=question,
                response=response,
            )
            relevancy = relevancy_evaluator.evaluate_response(
                query=question,
                response=response,
            )

            print(f"\nEvaluation question: {question}")
            print(
                "Faithfulness score:",
                faithfulness.score,
                "| passing:",
                faithfulness.passing,
            )
            print("Faithfulness feedback:", faithfulness.feedback)
            print(
                "Relevancy score:",
                relevancy.score,
                "| passing:",
                relevancy.passing,
            )
            print("Relevancy feedback:", relevancy.feedback)

        # Faithfulness measures whether the answer is supported by retrieved
        # context: 1.0 means supported, while 0.0 means the evaluator judged it
        # unsupported. Relevancy measures whether the response and retrieved
        # context address the query. They measure different qualities.
        #
        # Compare the two queries' scores after running. A question not answered
        # by the PDFs may get lower scores, but evaluator results can vary.
        # LLM-as-a-judge uses another language model to assess answers against
        # criteria. It helps when exact-answer accuracy is hard to define, but
        # its scores are judgments rather than guaranteed ground truth.

    except Exception as error:
        print(f"\nLlamaIndex pipeline could not complete: {error}")


if __name__ == "__main__":
    run_llamaindex_exercises()