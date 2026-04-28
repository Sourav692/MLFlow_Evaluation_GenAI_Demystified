# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 Lab 5.4b — Tool-Using Agent with LangGraph `create_react_agent`
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **LangGraph ReAct Agent** — Build a Reason → Act → Observe agent with `create_react_agent` and `@tool`-decorated functions
# MAGIC 2. **Auto-Traced Tools** — `mlflow.langchain.autolog()` emits TOOL + LLM spans automatically per invocation
# MAGIC 3. **Same Agent Scorers** — Reuse `ToolCallEfficiency` + `Correctness` + custom `tool_call_count` scorer from Lab 5.4
# MAGIC 4. **Multi-Turn LangGraph** — Pass full message history through the graph for `KnowledgeRetention` evaluation
# MAGIC 5. **v1 vs v2 Prompt Lift** — Quantify the path-quality improvement after constraining tool usage in the system prompt
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 5.4 (you understand `ToolCallEfficiency` and the trace-counting pattern)
# MAGIC - Lab 5.2b (you've used `mlflow.langchain.autolog()` once)
# MAGIC - `databricks-langchain` and `langgraph` installed (handled in Step 1)
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" databricks-langchain langgraph langchain-core

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Configure Namespace, Experiment, and Autolog
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 1 - CONFIGURE NAMESPACE, EXPERIMENT, AND AUTOLOG
# ============================================================================

import mlflow
from mlflow.entities import SpanType

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.langchain.autolog()

print(f"Experiment: {EXPERIMENT_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Define Tools with `@tool`
# MAGIC
# MAGIC LangChain's `@tool` decorator gives the function a JSON schema, a docstring used by the LLM as the tool description, and full integration with LangGraph's tool node. Each invocation auto-emits a `TOOL` span.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 2 - DEFINE TOOLS WITH `@TOOL`
# ============================================================================

from langchain_core.tools import tool

WEATHER = {"berlin": 12, "tokyo": 18, "san francisco": 16, "sydney": 22}
LOCAL_TIME = {"berlin": "14:05", "tokyo": "22:05", "san francisco": "05:05", "sydney": "23:05"}

@tool
def get_weather(city: str) -> dict:
    """Current temperature in Celsius for a city."""
    return {"city": city, "temp_c": WEATHER.get(city.lower(), 20)}

@tool
def get_time(city: str) -> dict:
    """Current local time for a city."""
    return {"city": city, "local_time": LOCAL_TIME.get(city.lower(), "12:00")}

TOOLS = [get_weather, get_time]
print("Tools registered:", [t.name for t in TOOLS])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Build the v1 LangGraph ReAct Agent
# MAGIC
# MAGIC `create_react_agent(model, tools, prompt=…)` returns a compiled graph implementing the **Reason → Act → Observe** loop:
# MAGIC
# MAGIC ```
# MAGIC START → llm → (tool_calls?) → tools → llm → … → END
# MAGIC ```
# MAGIC
# MAGIC State is automatically a list of `BaseMessage`. Each tool call adds a `ToolMessage` to state and routes back to the LLM until no more tool calls are emitted.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - BUILD THE V1 LANGGRAPH REACT AGENT
# ============================================================================

from databricks_langchain import ChatDatabricks
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatDatabricks(endpoint="databricks-claude-opus-4-6", temperature=0)

SYSTEM_PROMPT_V1 = (
    "You are a helpful assistant. Use tools when needed to answer the user."
)

agent_graph_v1 = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT_V1)

@mlflow.trace
def agent_v1_lg(question: str) -> str:
    state = agent_graph_v1.invoke({"messages": [HumanMessage(content=question)]})
    return state["messages"][-1].content

print(agent_v1_lg("What is the temperature in Berlin?"))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Tool-Use Eval Dataset (Same as Lab 5.4)
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 4 - TOOL-USE EVAL DATASET (SAME AS LAB 5.4)
# ============================================================================

import pandas as pd

tool_eval_rows = [
    {"inputs": {"question": "What is the temperature in Berlin?"},
     "expectations": {"expected_response": "12", "expected_tool_calls": 1}},
    {"inputs": {"question": "What time is it in Tokyo right now?"},
     "expectations": {"expected_response": "22:05", "expected_tool_calls": 1}},
    {"inputs": {"question": "Tell me the temperature in San Francisco."},
     "expectations": {"expected_response": "16", "expected_tool_calls": 1}},
    {"inputs": {"question": "What's the weather in Sydney?"},
     "expectations": {"expected_response": "22", "expected_tool_calls": 1}},
    {"inputs": {"question": "Give me the temperature in Berlin and the local time in Tokyo."},
     "expectations": {"expected_response": "Berlin 12, Tokyo 22:05", "expected_tool_calls": 2}},
]
tool_eval_df = pd.DataFrame(tool_eval_rows)
display(tool_eval_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Custom Trace-Based Tool-Count Scorer
# MAGIC
# MAGIC Identical scorer to Lab 5.4 — it counts `TOOL` spans regardless of which framework produced them. The only thing that needs to be true is that tool invocations carry `span_type=SpanType.TOOL`, which `mlflow.langchain.autolog()` guarantees.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 5 - CUSTOM TRACE-BASED TOOL-COUNT SCORER
# ============================================================================

from mlflow.entities import Feedback, AssessmentSource
from mlflow.genai.scorers import scorer

MAX_TOOL_CALLS_FOR_SIMPLE_Q = 3

@scorer
def tool_call_count(trace) -> Feedback:
    spans = trace.search_spans(span_type=SpanType.TOOL)
    n = len(spans)
    return Feedback(
        value="ok" if n <= MAX_TOOL_CALLS_FOR_SIMPLE_Q else "too_many",
        rationale=f"{n} tool calls (limit: {MAX_TOOL_CALLS_FOR_SIMPLE_Q})",
        source=AssessmentSource(source_type="CODE", source_id="tool_count_v1"),
    )

print("Tool-count scorer ready.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Evaluate v1 — `ToolCallEfficiency` + `Correctness` + Tool Count
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧮 STEP 6 - EVALUATE V1 - `TOOLCALLEFFICIENCY` + `CORRECTNESS` + TOOL COUNT
# ============================================================================

from mlflow.genai.scorers import ToolCallEfficiency, Correctness

results_v1 = mlflow.genai.evaluate(
    data=tool_eval_df,
    predict_fn=agent_v1_lg,
    scorers=[ToolCallEfficiency(), Correctness(), tool_call_count],
    model_id="models:/tool-agent-langgraph/v1",
)

display(results_v1.tables["eval_results"])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Multi-Turn Eval with `KnowledgeRetention`
# MAGIC
# MAGIC LangGraph stores conversation history in graph state. We pass a full message list and let the graph carry it through.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 7 - MULTI-TURN EVAL WITH `KNOWLEDGERETENTION`
# ============================================================================

@mlflow.trace
def multi_turn_agent_lg(messages: list[dict]) -> str:
    """Convert message dicts into LangChain messages and let the ReAct graph respond."""
    lc_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        # system turns are folded into the agent's prompt
    state = agent_graph_v1.invoke({"messages": lc_messages})
    return state["messages"][-1].content

multi_turn_rows = [
    {"inputs": {"messages": [
        {"role": "user", "content": "Hi! My name is Alex and I'm planning a trip to Tokyo."},
        {"role": "assistant", "content": "Nice to meet you, Alex! Tokyo's a great choice."},
        {"role": "user", "content": "What time is it there right now?"},
        {"role": "assistant", "content": "It's 22:05 in Tokyo."},
        {"role": "user", "content": "Cool — and remind me, what city did I say I was visiting?"},
    ]}},
    {"inputs": {"messages": [
        {"role": "user", "content": "I prefer temperatures in Fahrenheit."},
        {"role": "assistant", "content": "Got it — I'll use Fahrenheit."},
        {"role": "user", "content": "What's the temperature in Berlin?"},
    ]}},
]
multi_turn_df = pd.DataFrame(multi_turn_rows)


# COMMAND ----------

# ============================================================================
# ▶️ STEP
# ============================================================================

from mlflow.genai.scorers import KnowledgeRetention

results_multi = mlflow.genai.evaluate(
    data=multi_turn_df,
    predict_fn=multi_turn_agent_lg,
    scorers=[KnowledgeRetention()],
    model_id="models:/tool-agent-langgraph/v1-multiturn",
)

display(results_multi.tables["eval_results"])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Optimize the Prompt: v2 ReAct Agent
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 8 - OPTIMIZE THE PROMPT: V2 REACT AGENT
# ============================================================================

SYSTEM_PROMPT_V2 = (
    "You are a helpful assistant with access to weather and time tools. "
    "Rules:\n"
    "1. Call each tool at most ONCE per city per user turn.\n"
    "2. Do not call a tool if the user is making conversation, not asking for data.\n"
    "3. After tools return, answer concisely without re-checking."
)

agent_graph_v2 = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT_V2)

@mlflow.trace
def agent_v2_lg(question: str) -> str:
    state = agent_graph_v2.invoke({"messages": [HumanMessage(content=question)]})
    return state["messages"][-1].content

results_v2 = mlflow.genai.evaluate(
    data=tool_eval_df,
    predict_fn=agent_v2_lg,
    scorers=[ToolCallEfficiency(), Correctness(), tool_call_count],
    model_id="models:/tool-agent-langgraph/v2",
)

display(results_v2.tables["eval_results"])


# COMMAND ----------

# ============================================================================
# ▶️ SIDE-BY-SIDE: V1 VS V2 (LANGGRAPH REACT AGENT)
# ============================================================================

def behaviour_summary(results, label):
    return results.tables["eval_results"].selectExpr(
        f"'{label}' AS run",
        "AVG(CASE WHEN `tool_call_efficiency/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS efficiency_pass",
        "AVG(CASE WHEN `correctness/v1/value`         = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness_pass",
        "AVG(CASE WHEN `tool_call_count/v1/value`     = 'ok'  THEN 1.0 ELSE 0.0 END) AS within_call_budget",
    )

display(behaviour_summary(results_v1, "v1").union(behaviour_summary(results_v2, "v2")))


# COMMAND ----------

# ============================================================================
# ▶️ PER-ROW TOOL-CALL COUNTS
# ============================================================================

def per_row_counts(results, label):
    return results.tables["eval_results"].selectExpr(
        f"'{label}' AS run",
        "inputs",
        "`tool_call_count/v1/value` AS count_bucket",
        "`tool_call_count/v1/rationale` AS detail",
    )

display(per_row_counts(results_v1, "v1").union(per_row_counts(results_v2, "v2")))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9 — Compare Frameworks: Hand-Written Loop (5.4) vs LangGraph (5.4b)
# MAGIC
# MAGIC Open the experiment in the MLflow UI, select runs from both labs, and **Compare**. You should see:
# MAGIC - Comparable correctness (same tools, same model, equivalent prompts)
# MAGIC - Comparable tool-call counts on simple questions
# MAGIC - Possibly slightly different efficiency rationales — LangGraph's ReAct loop is a known good baseline
# MAGIC
# MAGIC **Reading the takeaway:** identical scorers + identical traces ⇒ honest framework-vs-framework comparison.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | LangGraph `create_react_agent` agent built with `@tool` functions | ✅ |
# MAGIC | `mlflow.langchain.autolog()` produces TOOL + LLM spans automatically | ✅ |
# MAGIC | `ToolCallEfficiency` + custom `tool_call_count` running on LangGraph traces | ✅ |
# MAGIC | `KnowledgeRetention` applied to multi-turn LangGraph invocation | ✅ |
# MAGIC | v1 vs v2 prompt comparison quantifies path-quality lift | ✅ |
# MAGIC
# MAGIC **Module 5 Outcome (LangGraph track):** You can build RAG and tool-using agents with LangGraph and evaluate them with the same MLflow harness used for raw-Python agents. Framework choice is an implementation detail — the eval contract is span types + `predict_fn` shape.
# MAGIC
# MAGIC Next: **Module 6** — Human review, labelling sessions, and judge calibration against human ground truth.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. `create_react_agent` in One Slide
# MAGIC - Returns a compiled graph implementing the ReAct loop: LLM → (tool calls?) → tools → LLM → … → END.
# MAGIC - State is a list of `BaseMessage`. Tool calls become `ToolMessage` entries automatically.
# MAGIC - Add a system prompt via the `prompt=` argument — that's the only knob most prompt iterations touch.
# MAGIC
# MAGIC ### 2. Why the Eval Code Doesn't Change
# MAGIC - `tool_call_count` counts `SpanType.TOOL` spans — LangGraph's autolog produces those by default.
# MAGIC - `ToolCallEfficiency` reads the same trace shape from any framework.
# MAGIC - `KnowledgeRetention` only needs a multi-turn message log — also framework-agnostic.
# MAGIC
# MAGIC ### 3. Framework-vs-Framework Comparison
# MAGIC - Run Lab 5.4 (raw OpenAI loop) and 5.4b (LangGraph) in the same experiment, then Compare.
# MAGIC - Differences usually reflect *prompt* differences, not framework choice.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Module 6** — Human review, labelling sessions, and judge calibration against human ground truth.
# MAGIC