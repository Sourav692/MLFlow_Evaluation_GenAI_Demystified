# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 5.4 — Agent-Specific Judges: Tool Call Efficiency & Knowledge Retention
# MAGIC
# MAGIC **Goal:** Evaluate **agentic behaviour** — not just the final answer, but **how efficiently the agent reached it** and **whether it remembers context across turns**.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Built a small **tool-using agent** that can call `get_weather` and `get_time`
# MAGIC 2. Applied the built-in **`ToolCallEfficiency`** judge to detect redundant tool calls
# MAGIC 3. Written a **trace-based scorer** that counts `TOOL` spans and flags any single-fact query with > 3 tool calls
# MAGIC 4. Applied **`KnowledgeRetention`** to a multi-turn conversation
# MAGIC 5. **Compared v1 vs v2** of the agent — same tools, optimized system prompt — and quantified the improvement
# MAGIC
# MAGIC > **Why this lab matters:** A correct answer that took 8 tool calls when 1 would do is still a production problem (cost, latency, fragility). Final-answer judges miss this entirely. Agent judges look at the *path*, not just the destination.
# MAGIC
# MAGIC > **Prereq:** Lab 1.3, Lab 4.2/4.3 (judges + `@scorer`), and the trace-span concepts from Lab 5.3.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure Namespace, Experiment, and Autolog

# COMMAND ----------

import json
import mlflow
from mlflow.entities import SpanType

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_05"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.openai.autolog()

print(f"Experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Build a Tool-Using Agent (v1)
# MAGIC
# MAGIC Two tools, both deterministic so we can reason about *correct* behaviour:
# MAGIC - `get_weather(city)` → temperature in C
# MAGIC - `get_time(city)` → local time
# MAGIC
# MAGIC We trace the dispatcher loop with `@mlflow.trace`. Each tool invocation gets its own span via `span_type=SpanType.TOOL` so trace-based scorers can count them.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

llm_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

WEATHER = {"berlin": 12, "tokyo": 18, "san francisco": 16, "sydney": 22}
LOCAL_TIME = {"berlin": "14:05", "tokyo": "22:05", "san francisco": "05:05", "sydney": "23:05"}

@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": WEATHER.get(city.lower(), 20)}

@mlflow.trace(span_type=SpanType.TOOL)
def get_time(city: str) -> dict:
    return {"city": city, "local_time": LOCAL_TIME.get(city.lower(), "12:00")}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current temperature in C for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Current local time for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]

TOOL_REGISTRY = {"get_weather": get_weather, "get_time": get_time}

# v1 system prompt: vague — invites redundant calls
SYSTEM_PROMPT_V1 = (
    "You are a helpful assistant. Use tools when needed to answer the user."
)

@mlflow.trace
def agent_v1(question: str) -> str:
    return _run_agent(question, SYSTEM_PROMPT_V1, max_iters=8)

def _run_agent(question: str, system_prompt: str, max_iters: int = 8) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    for _ in range(max_iters):
        resp = llm_client.chat.completions.create(
            model="databricks-claude-sonnet-4",
            messages=messages,
            tools=TOOLS_SPEC,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = TOOL_REGISTRY[tc.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "(max tool iterations reached)"

print(agent_v1("What is the temperature in Berlin?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build a Tiny Tool-Use Eval Dataset
# MAGIC
# MAGIC Each row pins the question, the **expected answer**, and the **expected number of tool calls**. The tool-call count gives our custom scorer something concrete to compare against.

# COMMAND ----------

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
# MAGIC ## Step 4 — Run the Built-in `ToolCallEfficiency` Judge
# MAGIC
# MAGIC `ToolCallEfficiency` reads the trace, looks at each tool span, and judges whether the **set of calls was minimal and non-redundant** for the given question. No extra labelling needed — it's a behaviour judge over the trace.

# COMMAND ----------

from mlflow.genai.scorers import ToolCallEfficiency, Correctness

results_v1 = mlflow.genai.evaluate(
    data=tool_eval_df,
    predict_fn=agent_v1,
    scorers=[ToolCallEfficiency(), Correctness()],
    model_id="models:/tool-agent/v1",
)

display(results_v1.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Custom Trace-Based Scorer: TOOL-Span Counter
# MAGIC
# MAGIC `ToolCallEfficiency` gives a yes/no quality verdict. We complement it with a **deterministic counter** — useful for hard SLAs ("never make more than 3 tool calls for a single-fact question").

# COMMAND ----------

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

print("Custom tool-count scorer ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Multi-Turn Eval with `KnowledgeRetention`
# MAGIC
# MAGIC `KnowledgeRetention` checks whether the agent **remembers facts established earlier in a conversation**. It needs a multi-turn trace, so we build one explicitly: the user states a fact in turn 1 ("My name is Alex") and asks about it later.

# COMMAND ----------

@mlflow.trace
def multi_turn_agent(messages: list[dict]) -> str:
    """Pass-through agent that takes a full message history and returns the next assistant turn."""
    full = [{"role": "system", "content": SYSTEM_PROMPT_V1}] + messages
    resp = llm_client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=full,
        tools=TOOLS_SPEC,
    )
    return resp.choices[0].message.content or ""

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

from mlflow.genai.scorers import KnowledgeRetention

results_multi = mlflow.genai.evaluate(
    data=multi_turn_df,
    predict_fn=multi_turn_agent,
    scorers=[KnowledgeRetention()],
    model_id="models:/tool-agent/v1-multiturn",
)

display(results_multi.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Optimize the Prompt: Agent v2
# MAGIC
# MAGIC v1's vague prompt led the model to over-call tools (e.g., fetching the same city's weather twice "to be sure"). v2 adds explicit instructions:
# MAGIC - Call each tool **at most once per entity per turn**.
# MAGIC - Don't call a tool if the user's question doesn't need it.

# COMMAND ----------

SYSTEM_PROMPT_V2 = (
    "You are a helpful assistant with access to weather and time tools. "
    "Rules:\n"
    "1. Call each tool at most ONCE per city per user turn.\n"
    "2. Do not call a tool if the user is making conversation, not asking for data.\n"
    "3. After tools return, answer concisely without re-checking."
)

@mlflow.trace
def agent_v2(question: str) -> str:
    return _run_agent(question, SYSTEM_PROMPT_V2, max_iters=8)

results_v2 = mlflow.genai.evaluate(
    data=tool_eval_df,
    predict_fn=agent_v2,
    scorers=[ToolCallEfficiency(), Correctness(), tool_call_count],
    model_id="models:/tool-agent/v2",
)

display(results_v2.tables["eval_results"])

# COMMAND ----------

# DBTITLE 1,Side-by-side: v1 vs v2 on tool-use behaviour
def behaviour_summary(results, label):
    return results.tables["eval_results"].selectExpr(
        f"'{label}' AS run",
        "AVG(CASE WHEN `tool_call_efficiency/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS efficiency_pass",
        "AVG(CASE WHEN `correctness/v1/value`         = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness_pass",
    )

# v1 has no tool_call_count column; rerun v1 with the counter for a fair comparison.
results_v1_with_count = mlflow.genai.evaluate(
    data=tool_eval_df,
    predict_fn=agent_v1,
    scorers=[ToolCallEfficiency(), Correctness(), tool_call_count],
    model_id="models:/tool-agent/v1-with-count",
)

display(behaviour_summary(results_v1_with_count, "v1").union(behaviour_summary(results_v2, "v2")))

# COMMAND ----------

# DBTITLE 1,Per-row tool-call counts: v1 vs v2
def per_row_counts(results, label):
    return results.tables["eval_results"].selectExpr(
        f"'{label}' AS run",
        "inputs",
        "`tool_call_count/v1/value` AS count_bucket",
        "`tool_call_count/v1/rationale` AS detail",
    )

display(per_row_counts(results_v1_with_count, "v1").union(per_row_counts(results_v2, "v2")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Compare Runs in the MLflow UI
# MAGIC
# MAGIC Open the experiment, select the **v1** and **v2** runs, click **Compare**. You'll see:
# MAGIC - `tool_call_efficiency` pass rate ↑
# MAGIC - `tool_call_count` distribution shift toward `ok`
# MAGIC - `correctness` ideally unchanged (we're optimising path, not destination)
# MAGIC
# MAGIC If correctness drops with v2, the new prompt is too restrictive — iterate.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | Tool-using agent built; tool spans visible in trace | ✅ |
# MAGIC | `ToolCallEfficiency` applied to detect redundant calls | ✅ |
# MAGIC | Trace-based `tool_call_count` scorer flags > 3 calls | ✅ |
# MAGIC | `KnowledgeRetention` applied to multi-turn conversation | ✅ |
# MAGIC | v1 vs v2 system-prompt comparison quantified | ✅ |
# MAGIC
# MAGIC **Module 5 Outcome:** You can evaluate full RAG pipelines across all three quality dimensions and diagnose failure modes from traces. You can evaluate agent tool-use behaviour and compare agent versions quantitatively.
# MAGIC
# MAGIC Next: **Module 6** — Human review, labelling sessions, and judge calibration against human ground truth.
