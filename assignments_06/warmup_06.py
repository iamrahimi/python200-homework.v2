"""Week 6 warmups: RAG concepts, keyword retrieval, and LlamaIndex."""

from dotenv import load_dotenv
import os
from pathlib import Path
import string

# Load environment variables from .env
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---

# Concepts Q1
# Scenario A — RAG. The assistant should retrieve current passages from the
# policy PDFs at question time. Updating the source library then updates what
# the assistant can answer without retraining the model.
#
# Scenario B — Fine-tuning. The 3,000 examples can teach the model a consistent,
# unusual writing style. Prompt instructions can help too, but fine-tuning is a
# better fit when the desired voice needs to appear reliably across many outputs.
#
# Scenario C — Prompt engineering. For one short report, include its text in a
# carefully written prompt and ask the model to answer only from that report.
# Building a reusable retrieval system would add unnecessary setup.

# Concepts Q2
# A confident answer can sound trustworthy even when its details are invented,
# so people may act on it without checking. For example, a fabricated medication
# dosage could lead a patient to take an unsafe amount. A response that says it
# is unsure invites the person to verify the information; confident tone can
# hide uncertainty and make a wrong answer more persuasive.

# Concepts Q3 — Correct RAG pipeline order:
# 1. Extract text from source documents — read the files and obtain their text.
# 2. Split text into chunks — divide long text into smaller retrievable sections.
# 3. Convert text chunks into embeddings — represent each chunk as a vector.
# 4. Receive the user's query — accept the question to answer.
# 5. Embed the user's query — represent the question as a vector in the same space.
# 6. Retrieve the most relevant chunks — compare the query vector with chunk vectors.
# 7. Inject retrieved chunks into the prompt — give the LLM relevant source context.
# 8. Generate a response from the LLM — answer using the question and retrieved text.


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
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Keyword Question 1
query = "What are your hours on weekends?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# hours.txt is selected because its text overlaps query terms such as "hours"
# and "weekends"; the other documents do not describe opening times.

# Keyword Question 2
query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# The function returns "None found": none of the query's meaningful words occur
# exactly in the menu text. Keyword matching misses related terms such as coffee
# drinks; semantic retrieval could connect caffeine to espresso or cold brew.

# Keyword Question 3 — predict before checking:
# Prediction: "None found." The loyalty document describes a loyalty program
# and points, but it never uses the exact word "rewards" or "sign up".
query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# There is no positive overlap, so the function returns its no-match sentinel
# and prints "None found." The loyalty document would be a good semantic match,
# but this exact-word function cannot connect "rewards" with "loyalty program."


# --- Semantic RAG Concepts ---

# Semantic Question 1
# 1. An embedding is a list of numbers that captures some of the meaning of a
# piece of text. Similar ideas tend to have vectors that point in similar directions.
# 2. The chunk with 0.85 is more relevant here. Its meaning is much closer to the
# query than the chunk scoring 0.30; cosine similarity compares vector directions.
# 3. Semantic search compares meaning represented by vectors, so related phrases
# can match even when they use different words (for example, "car" and "vehicle").

# Semantic Question 2
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Query and chunk embeddings |
# | What is retrieved?         | Full document                     | Most similar text chunks |
# | Can it handle synonyms?    | No                                | Often, based on similar meaning |
# | Storage format             | Plain text dictionary             | Vector index/store with embeddings |
# | Relevance score            | Number of overlapping keywords    | Similarity score, often cosine similarity |


# --- LlamaIndex ---
# Set BRIGHTLEAF_PDF_DIR if your lesson PDFs are in a different location.
# The provided BrightLeaf PDFs are kept in assignments_06/resources/brightleaf_pdfs.
BRIGHTLEAF_PDF_DIR = Path(
    os.getenv(
        "BRIGHTLEAF_PDF_DIR",
        str(Path(__file__).resolve().parent / "resources" / "brightleaf_pdfs"),
    )
)


def show_retrieved_nodes(response):
    """Print each source chunk's score and text, as requested in the assignment."""
    for number, source_node in enumerate(response.source_nodes, start=1):
        score = source_node.score
        text = source_node.node.get_content().replace("\n", " ")
        print(f"  Source {number}: score={score}")
        print(f"    {text[:150]}")


def show_eval_result(label, result):
    print(f"{label} score: {result.score}")
    print(f"{label} feedback: {result.feedback}")


def run_llamaindex_exercises():
    """Build and query the PDF index. Runs only when key and PDF folder exist."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\nSkipping LlamaIndex: OPENAI_API_KEY is not set.")
        return
    if not BRIGHTLEAF_PDF_DIR.is_dir():
        print(f"\nSkipping LlamaIndex: PDF folder not found: {BRIGHTLEAF_PDF_DIR}")
        print("Set BRIGHTLEAF_PDF_DIR in your environment to the correct folder.")
        return

    try:
        from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
    except ImportError as error:
        print(f"\nSkipping LlamaIndex: install the assignment packages first ({error}).")
        return

    # Use the embedding model specified in the assignment before creating the index.
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    # Set an available generation model explicitly; otherwise this LlamaIndex
    # version may default to the deprecated gpt-3.5-turbo model.
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    documents = SimpleDirectoryReader(str(BRIGHTLEAF_PDF_DIR)).load_data()
    if not documents:
        print(f"No readable PDF documents found in {BRIGHTLEAF_PDF_DIR}")
        return
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # LlamaIndex Question 1
    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]
    for question in questions:
        response = query_engine.query(question)
        print(f"\nQuestion: {question}")
        print("Answer:", response)
        show_retrieved_nodes(response)
    # Q1 benefits-query observations, grounded in the supplied PDFs:
    # employee_benefits.pdf directly answers this question with health, vision,
    # wellness, retirement, parental-leave, flexibility, and career-growth details.
    # Retrieved chunks from that file are relevant; the answer can be specific
    # because the document gives concrete benefits and amounts. Check the printed
    # response for its exact tone and note any unexpected source nodes after your run.
    #
    # Q1 security-query observations, grounded in the supplied PDFs:
    # security_policy.pdf directly covers MFA, VPN, encryption, incident response,
    # training, access reviews, and physical safeguards. Chunks from that file are
    # relevant, and the response can be specific because the policy lists controls.
    # Check the printed answer's exact tone and note any unrelated nodes after your run.
    # These relevance notes describe the source contents; exact retrieved nodes
    # and model wording depend on running with your local API key and index.

    # LlamaIndex Question 2
    question = questions[0]
    for top_k in (1, 5):
        engine = index.as_query_engine(similarity_top_k=top_k)
        response = engine.query(question)
        print(f"\nQuestion (similarity_top_k={top_k}): {question}")
        print("Answer:", response)
        for source_node in response.source_nodes:
            print(f"  score={source_node.score}")
    # Compare these answers. More context can add useful details, but irrelevant
    # chunks can distract the model, so a larger top_k is not always better.

    # LlamaIndex Question 3
    difficult_question = "What is BrightLeaf's favorite lunch restaurant, and why?"
    response = query_engine.query(difficult_question)
    print(f"\nQuestion (likely not covered): {difficult_question}")
    print("Answer:", response)
    show_retrieved_nodes(response)
    # Expected: the documents probably do not mention a favorite restaurant.
    # The retriever may still return the closest available chunks. A production
    # system should set a relevance threshold and instruct the LLM to say when the
    # sources do not contain the answer; query clarification can help with vague asks.

    # LlamaIndex Question 4 — LLM-as-a-judge evaluation.
    judge_llm = OpenAI(model="gpt-4o-mini")
    faithfulness = FaithfulnessEvaluator(llm=judge_llm)
    relevancy = RelevancyEvaluator(llm=judge_llm)

    q = "What employee benefits does BrightLeaf offer?"
    answer = query_engine.query(q)
    print(f"\nEvaluation query: {q}")
    show_eval_result("Faithfulness", faithfulness.evaluate_response(query=q, response=answer))
    show_eval_result("Relevancy", relevancy.evaluate_response(query=q, response=answer))

    low_quality_query = "What is BrightLeaf's office Wi-Fi password?"
    low_quality_answer = query_engine.query(low_quality_query)
    print(f"\nEvaluation query expected to be unsupported: {low_quality_query}")
    show_eval_result(
        "Faithfulness",
        faithfulness.evaluate_response(query=low_quality_query, response=low_quality_answer),
    )
    show_eval_result(
        "Relevancy",
        relevancy.evaluate_response(query=low_quality_query, response=low_quality_answer),
    )
    # A faithfulness score of 1.0 means the judge found the answer supported by
    # the retrieved context; 0.0 means it found no support (or judged it unfaithful).
    # Relevancy measures whether the answer addresses the user's question. It can
    # differ from faithfulness: an answer may use the context accurately but fail
    # to answer the question, or answer the question while adding unsupported claims.
    # The scores may drop for the Wi-Fi question because that private detail is
    # unlikely to appear in the documents. LLM-as-a-judge uses another language
    # model to assess qualities such as support and relevance, which are harder to
    # measure with exact-match accuracy when many valid answers use different words.


if __name__ == "__main__":
    run_llamaindex_exercises()
