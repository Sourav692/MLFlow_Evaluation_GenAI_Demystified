# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 3.2 — Pattern 1: Plain Python Function (Local Agent)
# MAGIC
# MAGIC **Goal:** Wire up a locally-defined agent function to `mlflow.genai.evaluate()`. This is the most common pattern during development — fastest iteration loop, no deployment needed.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Built a simple Q&A agent using the `databricks-openai` client in this notebook
# MAGIC 2. Decorated it with `@mlflow.trace` to satisfy the trace requirement
# MAGIC 3. Defined `predict_fn(question: str)` — the parameter name **must match** the dataset's `inputs.question` key
# MAGIC 4. Run `mlflow.genai.evaluate(data=..., predict_fn=my_agent, scorers=[...])`
# MAGIC 5. Verified each row produced a trace in the MLflow UI linked to the evaluation run
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 + Lab 2.2 done — the `tutorial_eval_v1` dataset must exist.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure Namespace and Experiment
# MAGIC
# MAGIC We reuse the catalog, schema, and experiment from Modules 1 and 2. The eval results land in this experiment alongside the traces produced by `@mlflow.trace`.

# COMMAND ----------

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_01"
DATASET_FQN = f"{CATALOG}.{SCHEMA}.tutorial_eval_v1"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"Experiment: {EXPERIMENT_PATH}")
print(f"Eval data:  {DATASET_FQN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Load the Eval Dataset
# MAGIC
# MAGIC `mlflow.genai.datasets.get_dataset(...)` returns the dataset object built in Lab 2.2. We pass `eval_dataset` directly into `evaluate()` later — MLflow understands the schema natively.

# COMMAND ----------

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)
print(f"Loaded {eval_dataset.to_df().count()} rows from {eval_dataset.name}")
display(eval_dataset.to_df().limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build the Local Agent
# MAGIC
# MAGIC Three things must be true for `evaluate()` to call this function correctly:
# MAGIC 1. **Decorated with `@mlflow.trace`** — every invocation produces a trace, which scorers can inspect for retrieval/tool-call metadata.
# MAGIC 2. **Parameter name matches the dataset key.** The dataset has `inputs.question`, so the function signature must be `def my_agent(question: str)`. MLflow passes each row's inputs as keyword arguments.
# MAGIC 3. **Returns a string (or dict).** A plain string becomes the `response` column in the results.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

@mlflow.trace
def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": "You are a Databricks expert. Answer concisely in 2-3 sentences."},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content

# Quick smoke test
print(my_agent("What is Delta Live Tables?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Run `mlflow.genai.evaluate`
# MAGIC
# MAGIC We score with two built-in judges:
# MAGIC - **`Correctness`** — checks the response covers the dataset's `expected_facts`.
# MAGIC - **`RelevanceToQuery`** — checks the response actually answers the question (catches off-topic verbosity).
# MAGIC
# MAGIC Set `model_id` so eval results are versioned — re-running with a new prompt/model is one parameter change.

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RelevanceToQuery

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[Correctness(), RelevanceToQuery()],
    model_id="models:/my-agent/1",
)

print("Evaluation complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Inspect Results
# MAGIC
# MAGIC Each row in `eval_results` carries:
# MAGIC - the original `inputs` and `expectations`
# MAGIC - the agent's `response`
# MAGIC - one column per scorer with the verdict and rationale
# MAGIC - a `trace_id` linking back to the trace produced during evaluation

# COMMAND ----------

display(results.tables["eval_results"])

# COMMAND ----------

# DBTITLE 1,Aggregate scorer pass rate
agg = results.tables["eval_results"]
display(agg.selectExpr(
    "AVG(CASE WHEN `correctness/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness_pass_rate",
    "AVG(CASE WHEN `relevance_to_query/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS relevance_pass_rate",
    "COUNT(*) AS total_rows",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Verify Traces in the MLflow UI
# MAGIC
# MAGIC Open the experiment in the MLflow UI and switch to the **Evaluations** tab. You should see:
# MAGIC - One eval run grouped by `model_id`
# MAGIC - Each row click-through opens the trace from `@mlflow.trace`
# MAGIC - Scorer verdicts inline next to each response
# MAGIC
# MAGIC You can also fetch the linked traces programmatically:

# COMMAND ----------

traces = mlflow.search_traces(
    experiment_names=[EXPERIMENT_PATH],
    max_results=5,
    order_by=["attributes.timestamp_ms DESC"],
)
display(traces)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | Local agent decorated with `@mlflow.trace` | ✅ |
# MAGIC | `predict_fn` parameter name matches `inputs.question` | ✅ |
# MAGIC | `evaluate()` ran with Correctness + RelevanceToQuery | ✅ |
# MAGIC | Each row produced a linked trace in the UI | ✅ |
# MAGIC | `model_id` set so the run is versioned | ✅ |
# MAGIC
# MAGIC Next: **Lab 3.3** — switch from a local function to a deployed Model Serving endpoint with no code changes to the endpoint itself.
