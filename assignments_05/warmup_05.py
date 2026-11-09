import json

from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from the environment (including a local .env file)
# before creating the client or making any API calls.
load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"


def get_text(response):
    """Return the text content from the first choice in a chat response."""
    return response.choices[0].message.content or ""


# --- Completions API ---

# API Q1
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)
print("API Q1 response:", get_text(response))
print("API Q1 model:", response.model)
print("API Q1 total tokens:", response.usage.total_tokens)


# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(f"API Q2 response (temperature={temperature}): {get_text(response)}")

# Observation: Lower temperatures tend to produce more predictable wording, while
# higher temperatures can produce more varied or surprising names. For a
# consistent, reproducible result, I would use temperature=0 (though an API call
# may not be perfectly identical in every circumstance).


# API Q3
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)
for index, choice in enumerate(response.choices, start=1):
    print(f"API Q3 completion {index}:", choice.message.content)


# API Q4
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work in a detailed, beginner-friendly way.",
        }
    ],
    max_tokens=15,
)
print("API Q4 response (limited to 15 output tokens):", get_text(response))
# The response is cut off when it reaches the token limit. max_tokens can help
# control response length, latency, and cost in an application.


# --- System Messages and Personas ---

# System Q1
messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]
response = client.chat.completions.create(model=MODEL, messages=messages)
print("System Q1 patient tutor:", get_text(response))

messages = [
    {
        "role": "system",
        "content": "You are a dramatic pirate captain who explains programming with playful sea metaphors.",
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]
response = client.chat.completions.create(model=MODEL, messages=messages)
print("System Q1 pirate captain:", get_text(response))
# The tutor uses a patient, encouraging tone; the pirate uses playful nautical
# metaphors and a more dramatic voice.


# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]
response = client.chat.completions.create(model=MODEL, messages=messages)
print("System Q2 response:", get_text(response))
# The model has no memory between calls, but this call includes the earlier
# messages as context, so it can read Jordan's name from the conversation.


# --- Prompt Engineering ---

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]


def classify_reviews(instructions):
    """Ask the model to classify the three shared reviews."""
    review_text = "\n".join(
        f'Review {index}: "{review}"' for index, review in enumerate(reviews, start=1)
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"{instructions}\n\n{review_text}",
            }
        ],
    )
    return get_text(response)


def print_classifications(label, result):
    """Print each review's classification with its review number."""
    print(f"{label} results:")
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    for index, line in enumerate(lines, start=1):
        print(f"Review {index} result: {line}")


# Prompt Q1 — Zero-Shot
zero_shot_result = classify_reviews(
    "Classify the sentiment of each review as positive, negative, or mixed. "
    "Return one line per review using the format 'Review N: sentiment'."
)
print_classifications("Prompt Q1 zero-shot", zero_shot_result)


# Prompt Q2 — One-Shot
one_shot_result = classify_reviews(
    'Classify the sentiment of each review as positive, negative, or mixed. '
    'Return one line per review using the format "Review N: sentiment".\n'
    'Example:\nReview: "Fast shipping but the item arrived damaged."\nSentiment: mixed'
)
print_classifications("Prompt Q2 one-shot", one_shot_result)
# Adding an example can clarify the requested format and encourage consistent
# labels, especially when the example demonstrates the expected output style.


# Prompt Q3 — Few-Shot
few_shot_result = classify_reviews(
    'Classify the sentiment of each review as positive, negative, or mixed. '
    'Return one line per review using the format "Review N: sentiment".\n'
    'Examples:\n'
    'Review: "The staff was kind and the service was excellent."\nSentiment: positive\n'
    'Review: "The product broke immediately and no one helped me."\nSentiment: negative\n'
    'Review: "The food was delicious, but the wait was too long."\nSentiment: mixed'
)
print_classifications("Prompt Q3 few-shot", few_shot_result)
# Zero-shot is useful for a quick, straightforward task. One-shot can show the
# desired output format. Few-shot is helpful when several examples are needed
# to clarify labels or handle nuanced cases consistently.


# Prompt Q4 — Chain of Thought
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": (
                "Solve this salary problem. Show the calculation steps briefly, "
                "then clearly label the final answer. A data engineer earns "
                "$85,000 per year. She gets a 12% raise, then 6 months later "
                "takes a new job that pays $7,500 more per year than her "
                "post-raise salary. What is her final annual salary?"
            ),
        }
    ],
)
print("Prompt Q4 full response:", get_text(response))
# Asking for explicit calculation steps helps make the intermediate arithmetic
# visible, which can make a multi-step answer easier to check for mistakes.


# Prompt Q5 — Structured Output
review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": (
                "Analyze this review and return only valid JSON with exactly these keys: "
                '"sentiment", "confidence" (a number from 0 to 1), and '
                '"reason" (one sentence). Do not include markdown fences.\n'
                f"Review: {review}"
            ),
        }
    ],
)
raw_json = get_text(response)
print("Prompt Q5 raw response:", raw_json)
try:
    result = json.loads(raw_json)
    print("Prompt Q5 sentiment:", result["sentiment"])
    print("Prompt Q5 confidence:", result["confidence"])
    print("Prompt Q5 reason:", result["reason"])
except (json.JSONDecodeError, KeyError, TypeError) as error:
    print("Prompt Q5 JSON parsing failed:", error)
    print("Prompt Q5 raw response for debugging:", raw_json)


# Prompt Q6 — Delimiters
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": prompt}]
)
print("Prompt Q6 instructions passage:", get_text(response))

prose_text = "The mountain village sits beside a quiet lake. In autumn, the trees turn bright shades of red and gold."
prose_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{prose_text}```
"""
response = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": prose_prompt}]
)
print("Prompt Q6 prose passage:", get_text(response))
# Delimiters make it clear which text is input data and which text is the
# instruction, helping prevent instructions inside the data from being confused
# with the task itself.


# --- Local Models with Ollama ---

# Ollama Q1
# Run in a terminal:
# In two sentences, I can explain: "Large language models are AI systems that process and generate human-like text, trained on massive datasets to 
# understand and create language. They can understand complex patterns and generate text, making them powerful for tasks like writing or answering 
# questions."

# """
# Ollama Q1 OpenAI response: A large language model (LLM) is an artificial intelligence system designed to understand and generate human-like text by analyzing vast amounts of written data. 
# These models utilize deep learning techniques, specifically neural networks, to predict and produce coherent and contextually relevant language based on the input they receive.
# """

# My observations:
# I noticed that both responses explain that large language models learn from text
# and generate human-like responses. The first response is shorter and gives examples
# of what LLMs can do, while the OpenAI response explains a little more about how
# they work, including neural networks. One advantage of running a model locally is
# that I can use it without sending my prompts to an online service. One disadvantage
# is that it can use a lot of computer memory and may not be as powerful as an online model.

ollama_comparison_prompt = (
    "Explain what a large language model is in two sentences."
)

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": ollama_comparison_prompt}],
)
openai_response = get_text(response)
print("Ollama Q1 OpenAI response:", openai_response)

# After comparing the two outputs, replace these notes with your observations:
# My observations: [Describe differences in wording, detail, or speed.]
# Local advantage: [Write one advantage of running a model locally.]
# Local disadvantage: [Write one disadvantage of running a model locally.]