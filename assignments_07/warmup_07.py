"""Week 7 warmups: tool calling, ReAct, and smolagents.

Run this file from inside assignments_07/ after copying
resources/bike_commute.csv from the course repository.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Agents may create plots from worker threads. Use Matplotlib's non-GUI backend.
os.environ.setdefault("MPLBACKEND", "Agg")


# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


CELSIUS_TO_FAHRENHEIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a temperature from Celsius to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in degrees Celsius.",
                }
            },
            "required": ["celsius"],
            "additionalProperties": False,
        },
    },
}


def get_current_time() -> str:
    """Return the current local date and time as an ISO-formatted string."""
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


GET_CURRENT_TIME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def create_openai_client():
    """Load the .env file and return an OpenAI client when a key is available."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing; skipping API-dependent questions.")
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def call_with_tools(client, messages, tools):
    """Make one Chat Completions request with the provided JSON tool schemas."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )


def run_agent(query: str, client=None) -> str:
    """Week 2 agent: get_current_time is its only tool."""
    if client is None:
        client = create_openai_client()
    if client is None:
        return "No API response: set OPENAI_API_KEY in .env and run again."

    messages = [{"role": "user", "content": query}]
    response = call_with_tools(client, messages, [GET_CURRENT_TIME_SCHEMA])
    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        return assistant_message.content or ""

    messages.append(assistant_message)
    for tool_call in assistant_message.tool_calls:
        if tool_call.function.name == "get_current_time":
            result = get_current_time()
        else:
            result = f"Unknown tool: {tool_call.function.name}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        )

    final_response = call_with_tools(client, messages, [GET_CURRENT_TIME_SCHEMA])
    return final_response.choices[0].message.content or ""


def run_agent_with_temperature(query: str, client=None) -> str:
    """Agent with both time and Celsius conversion tools."""
    if client is None:
        client = create_openai_client()
    if client is None:
        return "No API response: set OPENAI_API_KEY in .env and run again."

    tools = [GET_CURRENT_TIME_SCHEMA, CELSIUS_TO_FAHRENHEIT_SCHEMA]
    messages = [{"role": "user", "content": query}]

    # Continue the ReAct loop until the assistant replies without requesting tools.
    for _ in range(6):
        response = call_with_tools(client, messages, tools)
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        messages.append(assistant_message)
        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            if name == "get_current_time":
                result = get_current_time()
            elif name == "celsius_to_fahrenheit":
                result = celsius_to_fahrenheit(float(arguments["celsius"]))
            else:
                result = f"Unknown tool: {name}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )
    return "The agent reached the tool-round limit before giving a final answer."


# --- Lesson 03: Multi-Tool Agent ---

# Q4
class CsvManager:
    """Load a CSV file and provide small data operations for the agents."""

    def __init__(self, resources_dir=None, outputs_dir=None):
        # Anchor defaults to this script so the paths work even if it is run
        # from the repository root instead of from inside assignments_07/.
        assignment_dir = Path(__file__).resolve().parent
        self.resources_dir = Path(resources_dir) if resources_dir else assignment_dir / "resources"
        self.outputs_dir = Path(outputs_dir) if outputs_dir else assignment_dir / "outputs"
        self.dataframe = None
        self.csv_path = None
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str) -> dict:
        """Load a CSV from resources/ and return its dimensions and columns."""
        import pandas as pd

        path = self.resources_dir / filename
        if not path.is_file():
            return {"error": f"CSV not found: {path}. Copy the course bike_commute.csv into resources/."}
        self.dataframe = pd.read_csv(path)
        self.csv_path = path
        return {
            "filename": path.name,
            "rows": int(self.dataframe.shape[0]),
            "columns": int(self.dataframe.shape[1]),
            "column_names": list(self.dataframe.columns),
        }

    def list_columns(self) -> list:
        """Return the loaded CSV's column names."""
        if self.dataframe is None:
            return ["error: no CSV is loaded"]
        return list(self.dataframe.columns)

    def get_summary(self) -> dict:
        """Return descriptive statistics for the loaded CSV."""
        if self.dataframe is None:
            return {"error": "No CSV is loaded."}
        return self.dataframe.describe(include="all").to_dict()

    def compute_correlation(self, col1: str, col2: str):
        """Compute the Pearson correlation between two columns in the loaded DataFrame.

        Returns the correlation coefficient and p-value.
        """
        if self.dataframe is None:
            return {"error": "No CSV is loaded."}
        if col1 not in self.dataframe.columns or col2 not in self.dataframe.columns:
            missing = [name for name in (col1, col2) if name not in self.dataframe.columns]
            return {"error": f"Column(s) not found: {', '.join(missing)}"}

        from scipy.stats import pearsonr

        try:
            pearson_r, p_value = pearsonr(self.dataframe[col1], self.dataframe[col2])
        except (TypeError, ValueError) as error:
            return {"error": f"Could not compute correlation: {error}"}
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }

    def plot_scatter(self, x_column: str, y_column: str) -> dict:
        """Save a scatter plot for two loaded columns and return its output path."""
        if self.dataframe is None:
            return {"error": "No CSV is loaded."}
        missing = [name for name in (x_column, y_column) if name not in self.dataframe.columns]
        if missing:
            return {"error": f"Column(s) not found: {', '.join(missing)}"}

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.outputs_dir / f"{x_column}_vs_{y_column}_scatter.png"
        plt.figure(figsize=(8, 5))
        # The fixed blue color is an intentional limit of this approved tool.
        plt.scatter(self.dataframe[x_column], self.dataframe[y_column], color="blue")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{y_column} vs {x_column}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return {"message": "Scatter plot saved.", "path": str(output_path)}


SYSTEM_PROMPT = """You are a helpful data-analysis assistant. Use the available tools to load and inspect CSV files, then answer questions using the results. Explain what you did briefly. Do not invent data or claim a tool succeeded if it returned an error."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "CSV filename in resources/."}},
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List the columns in the currently loaded CSV.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "Return descriptive statistics for the currently loaded CSV.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation and p-value for two columns in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "Name of the first numeric column."},
                    "col2": {"type": "string", "description": "Name of the second numeric column."},
                },
                "required": ["col1", "col2"],
                "additionalProperties": False,
            },
        },
    },
]


def run_agent_cycle(messages, query, csv_manager, client=None, max_tool_rounds=8):
    """Run a ReAct cycle, appending user, assistant, and tool messages in place."""
    if client is None:
        client = create_openai_client()
    if client is None:
        return "No API response: set OPENAI_API_KEY in .env and run again."

    node_tools = {
        "load_csv": csv_manager.load_csv,
        "list_columns": csv_manager.list_columns,
        "get_summary": csv_manager.get_summary,
        "compute_correlation": csv_manager.compute_correlation,
    }
    messages.append({"role": "user", "content": query})

    for _ in range(max_tool_rounds):
        response = call_with_tools(client, messages, TOOLS_SCHEMA)
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            final_text = assistant_message.content or ""
            messages.append({"role": "assistant", "content": final_text})
            return final_text

        messages.append(assistant_message)
        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            if name not in node_tools:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = node_tools[name](**arguments)
                except Exception as error:
                    result = {"error": f"{type(error).__name__}: {error}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(result, default=str),
                }
            )
    return "The agent reached the tool-round limit before giving a final answer."


# --- Lesson 04: smolagents ---

# Q7-Q8 tools and agents are created in main(), after the shared CsvManager exists.


def run_smolagents_questions(csv_manager):
    """Define the smolagents tools and run the Q7-Q8 prompts."""
    try:
        from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool
    except ImportError:
        print("Install smolagents with: pip install smolagents")
        return

    @tool
    def compute_correlation(col1: str, col2: str) -> dict:
        """Compute Pearson correlation and p-value for two numeric CSV columns.

        Args:
            col1: Name of the first numeric column in the loaded CSV.
            col2: Name of the second numeric column in the loaded CSV.
        """
        return csv_manager.compute_correlation(col1, col2)

    print(compute_correlation.description)
    # smolagents builds a tool description/schema from the function name, type
    # hints, and docstring. In the manual Q4 schema we supplied those fields
    # explicitly; clear parameter names and a useful docstring help it generate
    # an accurate description.

    @tool
    def load_csv(filename: str) -> dict:
        """Load a CSV file from the resources directory and return its columns.

        Args:
            filename: Name of the CSV file inside resources/.
        """
        return csv_manager.load_csv(filename)

    @tool
    def list_columns() -> list:
        """Return column names from the currently loaded CSV."""
        return csv_manager.list_columns()

    @tool
    def plot_scatter(x_column: str, y_column: str) -> dict:
        """Plot two columns as a blue scatter plot and save the image under outputs/.

        Args:
            x_column: Name of the column to use for the horizontal axis.
            y_column: Name of the column to use for the vertical axis.
        """
        return csv_manager.plot_scatter(x_column, y_column)

    tools = [load_csv, list_columns, compute_correlation, plot_scatter]
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing; skipping smolagents questions.")
        return

    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        api_key=api_key,
    )
    tool_agent = ToolCallingAgent(tools=tools, model=model, max_steps=8)
    code_agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=8,
        additional_authorized_imports=["matplotlib", "matplotlib.pyplot", "pandas"],
    )

    # Q8
    prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
    response_tool = tool_agent.run(prompt)
    print("ToolCallingAgent response:", response_tool)
    response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
    print("CodeAgent response:", response_code)

    # The ToolCallingAgent uses the plot tool, which only supports x/y columns
    # and fixes the dots to blue, so it cannot honor the requested green color.
    # The CodeAgent can write its own plotting code and can set the dots green.
    # This shows that tool agents fit bounded, approved actions, while code
    # agents fit flexible tasks that need operations not exposed by a tool.


# Q9 reflection:
# A ToolCallingAgent is a better choice for a task such as loading a report and
# returning a fixed summary. The allowed actions are known in advance, so the
# agent can choose from small, tested tools with clear inputs and outputs.
# A CodeAgent can generate and execute code. Incorrect or unsafe generated code
# could change files, consume resources, or access data beyond the intended
# analysis; a ToolCallingAgent is limited to the tools the developer provides.


if __name__ == "__main__":
    # Q1: call the converter directly, without an agent.
    for temperature in (0, 100, -40):
        print(celsius_to_fahrenheit(temperature))

    client = create_openai_client()
    if client is not None:
        # Q2 prediction: I expect no tool call. The request asks for a
        # calculation, but the only available tool returns the current time.
        # I expect one API call: the model can calculate and answer directly.
        response = run_agent("Convert 100 degrees Celsius to Fahrenheit", client)
        print("Q2 response:", response)
        # I can compare this response with the prediction above; the lesson
        # demonstrates that tool availability alone does not force a tool call.

        # Q3
        response_a = run_agent_with_temperature("What is 37 degrees Celsius in Fahrenheit?", client)
        print("Response A:", response_a)
        # Expected: celsius_to_fahrenheit is called because the query requests
        # the exact conversion the tool was designed to perform.

        response_b = run_agent_with_temperature("What is the boiling point of water in plain English?", client)
        print("Response B:", response_b)
        # Expected: no tool call, because the question is general knowledge and
        # does not ask to convert a supplied Celsius value.

    # Q4-Q6
    csv_manager = CsvManager()
    if client is not None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        result = run_agent_cycle(
            messages,
            "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
            csv_manager,
            client=client,
        )
        print(result)
        # In the ReAct loop, system gives instructions, user asks the task,
        # assistant reasons and may request a tool, and tool returns the result
        # for the assistant to use in its next response.
        print(json.dumps(messages, indent=2, default=str))

    # Q7-Q8
    run_smolagents_questions(csv_manager)
