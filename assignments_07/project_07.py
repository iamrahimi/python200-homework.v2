"""Week 7 mini-project: conversational World Happiness data agent."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

# CodeAgent may create plots in its execution environment. Avoid a GUI backend,
# which can fail on macOS when code runs in a worker thread.
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "resources" / "happiness_project"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Shared data frame used by the tools.
df = None

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",  # The 2024 file uses this name.
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load and combine the yearly World Happiness CSV files into memory.

    The files are read from happiness_project/ using semicolon separators and
    comma decimal marks. A year column is added from each filename, and common
    columns are renamed to country, region, happiness_score, and gdp_per_capita.

    Returns:
        A dictionary containing the combined data frame's shape and column names,
        or an error message if no yearly files are found.
    """
    global df

    csv_paths = sorted(DATA_DIR.glob("world_happiness_*.csv"))
    if not csv_paths:
        return {
            "error": (
                f"No yearly World Happiness CSV files found in {DATA_DIR}. "
                "Copy happiness_project/ into assignments_07/."
            )
        }

    frames = []
    for path in csv_paths:
        try:
            year_df = pd.read_csv(path, sep=";", decimal=",")
        except Exception as error:
            return {"error": f"Could not read {path.name}: {error}"}

        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])
        frames.append(year_df)

    df = pd.concat(frames, ignore_index=True)
    return {"shape": df.shape, "columns": list(df.columns)}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one column in the loaded dataset.

    Args:
        column: Exact name of the column to summarize, such as happiness_score
            or gdp_per_capita.

    Returns:
        Descriptive statistics as a dictionary, or an error if data is not
        loaded or the requested column does not exist.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}

    try:
        return df[column].describe().to_dict()
    except Exception as error:
        return {"error": f"Could not summarize {column}: {error}"}


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation and its p-value for two numeric columns.

    Args:
        col1: Name of the first numeric column, such as gdp_per_capita.
        col2: Name of the second numeric column, such as happiness_score.

    Returns:
        A dictionary with the column names, Pearson r, and p-value rounded to
        four decimal places, or an error message for invalid input.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}
    missing = [column for column in (col1, col2) if column not in df.columns]
    if missing:
        return {"error": f"Column(s) not found: {', '.join(missing)}"}

    try:
        values = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(values) < 2:
            return {"error": "At least two rows with numeric values are required."}
        pearson_r, p_value = pearsonr(values[col1], values[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }
    except Exception as error:
        return {"error": f"Could not compute correlation: {error}"}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries for a column in one year.

    Args:
        column: Column used to rank countries, for example happiness_score.
        year: Year to filter, from 2015 through 2024 when data is available.
        n: Number of leading countries to return. Defaults to 5.

    Returns:
        A dictionary containing a list of country names and their values, or an
        error message if the data, column, year, or requested number is invalid.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    if "country" not in df.columns:
        return {"error": "The dataset does not contain the country column."}
    if n < 1:
        return {"error": "n must be at least 1."}

    year_rows = df[df["year"] == year]
    if year_rows.empty:
        return {"error": f"No data found for year {year}."}
    try:
        ranked = year_rows.dropna(subset=[column]).sort_values(column, ascending=False).head(n)
        return {
            "year": int(year),
            "ranking_column": column,
            "countries": [
                {"country": str(row["country"]), column: float(row[column])}
                for _, row in ranked.iterrows()
            ],
        }
    except Exception as error:
        return {"error": f"Could not rank countries by {column}: {error}"}


SYSTEM_PROMPT = """You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
The yearly CSV files are in happiness_project/ beside this script. The plotting output
folder is outputs/. For custom analyses or plots, load and combine the CSV files in code
as needed; the tools return summaries and rankings, not the full DataFrame. After the
first query, the real combined DataFrame is also provided as the variable happiness_df.
Use happiness_df directly for custom analysis. Never create a DataFrame from the
load_happiness_data result: that tool returns only shape and column-name metadata.
Be concise and student-friendly in your responses."""


QUERIES = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


def main():
    """Configure the model and run the guided and student-created queries."""
    load_dotenv()
    # Keep relative paths used by the agent anchored to assignments_07 even if
    # this script was launched from the repository root.
    os.chdir(PROJECT_DIR)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your .env file before running the project.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
    agent = CodeAgent(
        tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
        model=model,
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=[
            "pandas",
            "matplotlib",
            "matplotlib.pyplot",
            "scipy.stats",
            "glob",
            "pathlib",
        ],
        max_steps=8,
    )

    # Task 3: Run each guided query in sequence so the agent retains context.
    for query in QUERIES:
        print(f"\n--- Query: {query} ---")
        run_options = {}
        if df is not None:
            run_options["additional_args"] = {"happiness_df": df}
        response = agent.run(query, reset=False, **run_options)
        print(response)

    # Task 4, query 1: this should use the ranking tool.
    my_query_1 = "Show me the top 3 happiest countries in 2015."
    response_1 = agent.run(my_query_1, reset=False, additional_args={"happiness_df": df})
    print(f"\n--- My query 1: {my_query_1} ---")
    print(response_1)
    # This should trigger get_top_n_countries because the requested ranking is
    # exactly what that tool provides.

    # Task 4, query 2: the tools do not compare regional changes across years,
    # so this should require the agent to write code and analyze the CSV files.
    my_query_2 = (
        "Which region had the largest increase in average happiness score from "
        "2015 to 2024? Calculate the change and explain the result."
    )
    response_2 = agent.run(my_query_2, reset=False, additional_args={"happiness_df": df})
    print(f"\n--- My query 2: {my_query_2} ---")
    print(response_2)
    # This should require generated Python code because no tool calculates this
    # multi-year, region-level change.


if __name__ == "__main__":
    main()


# --- Reflection ---
#
# 1. Query 3 gives both Pearson r and a p-value. I would check that the response
#    describes the direction and strength using r, and uses the p-value to explain
#    whether the evidence is statistically significant. A very small p-value is
#    evidence against zero correlation, but it does not show that GDP causes
#    happiness or measure the practical size of the relationship by itself.
#
# 2. One capability that can surprise me is that the CodeAgent can write custom
#    plotting code for a regional trend even though the four tools do not include
#    a plotting function. I should still check the saved chart and the code's
#    data handling rather than assuming the generated result is correct.
#
# 3. A useful additional tool would be compare_country_trend(country, start_year,
#    end_year). It could return that country's happiness scores and change over
#    the selected years, helping answer questions about one country's progress.
