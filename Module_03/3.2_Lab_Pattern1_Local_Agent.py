# Databricks notebook source
# MAGIC %md
# MAGIC # 🧑‍💻 Lab 3.2 — Pattern 1: Plain Python Function (Local Agent)
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Local Agent Pattern** — Wire a locally-defined function into `mlflow.genai.evaluate()` — the fastest dev loop
# MAGIC 2. **`@mlflow.trace`** — Decorate the agent so each invocation produces a trace linked to the eval run
# MAGIC 3. **Signature Contract** — Use parameter names that match the dataset's `inputs.<key>` keys
# MAGIC 4. **Built-in Judges** — Score with `Correctness` and `RelevanceToQuery` from `mlflow.genai.scorers`
# MAGIC 5. **Versioned Runs** — Set `model_id` so the run is comparable across iterations
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 — workspace setup complete
# MAGIC - Lab 2.2 — `tutorial_eval_v1` dataset exists in `genai_eval_tutorial.module_01`
# MAGIC - Foundation Model API access for `databricks-claude-opus-4-6`
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" databricks-openai

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Configure Namespace and Experiment
# MAGIC
# MAGIC We reuse the catalog, schema, and experiment from Modules 1 and 2. The eval results land in this experiment alongside the traces produced by `@mlflow.trace`.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 1 - CONFIGURE NAMESPACE AND EXPERIMENT
# ============================================================================

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
# MAGIC ---
# MAGIC ## Step 2 — Load the Eval Dataset
# MAGIC
# MAGIC `mlflow.genai.datasets.get_dataset(...)` returns the dataset object built in Lab 2.2. We pass `eval_dataset` directly into `evaluate()` later — MLflow understands the schema natively.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 2 - LOAD THE EVAL DATASET
# ============================================================================

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)
print(f"Loaded {eval_dataset.to_df().count()} rows from {eval_dataset.name}")
display(eval_dataset.to_df().head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Build the Local Agent
# MAGIC
# MAGIC Three things must be true for `evaluate()` to call this function correctly:
# MAGIC 1. **Decorated with `@mlflow.trace`** — every invocation produces a trace, which scorers can inspect for retrieval/tool-call metadata.
# MAGIC 2. **Parameter name matches the dataset key.** The dataset has `inputs.question`, so the function signature must be `def my_agent(question: str)`. MLflow passes each row's inputs as keyword arguments.
# MAGIC 3. **Returns a string (or dict).** A plain string becomes the `response` column in the results.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - BUILD THE LOCAL AGENT
# ============================================================================

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
# MAGIC ---
# MAGIC ## Step 4 — Run `mlflow.genai.evaluate`
# MAGIC
# MAGIC We score with two built-in judges:
# MAGIC - **`Correctness`** — checks the response covers the dataset's `expected_facts`.
# MAGIC - **`RelevanceToQuery`** — checks the response actually answers the question (catches off-topic verbosity).
# MAGIC
# MAGIC Set `model_id` so eval results are versioned — re-running with a new prompt/model is one parameter change.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧮 STEP 4 - RUN `MLFLOW.GENAI.EVALUATE`
# ============================================================================

from mlflow.genai.scorers import Correctness, RelevanceToQuery

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[Correctness(), RelevanceToQuery()],
)

print("Evaluation complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Inspect Results
# MAGIC
# MAGIC Each row in `eval_results` carries:
# MAGIC - the original `inputs` and `expectations`
# MAGIC - the agent's `response`
# MAGIC - one column per scorer with the verdict and rationale
# MAGIC - a `trace_id` linking back to the trace produced during evaluation
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 5 - INSPECT RESULTS
# ============================================================================

display(results.tables["eval_results"])


# COMMAND ----------

# ============================================================================
# ▶️ AGGREGATE SCORER PASS RATE
# ============================================================================

import pandas as pd

agg = results.tables["eval_results"]
summary = pd.DataFrame([{
    "correctness_pass_rate": (agg["correctness/value"] == "yes").mean(),
    "relevance_pass_rate": (agg["relevance_to_query/value"] == "yes").mean(),
    "total_rows": len(agg),
}])
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Verify Traces in the MLflow UI
# MAGIC
# MAGIC Open the experiment in the MLflow UI and switch to the **Evaluations** tab. You should see:
# MAGIC - One eval run grouped by `model_id`
# MAGIC - Each row click-through opens the trace from `@mlflow.trace`
# MAGIC - Scorer verdicts inline next to each response
# MAGIC
# MAGIC You can also fetch the linked traces programmatically:
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 6 - VERIFY TRACES IN THE MLFLOW UI
# ============================================================================

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id
traces = mlflow.search_traces(
    experiment_ids=[experiment_id],
    max_results=5,
    order_by=["attributes.timestamp_ms DESC"],
)
display(traces)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
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
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. Pattern 1: Local Function
# MAGIC - **Best for active development.** No deployment, no versioning overhead — change the code and re-run.
# MAGIC - MLflow passes each row's `inputs` dict as keyword arguments — parameter names matter.
# MAGIC
# MAGIC ### 2. Tracing Contract
# MAGIC - **`@mlflow.trace` is non-negotiable.** Without it, scorers that inspect retrieval / tool calls have nothing to read.
# MAGIC - Each evaluated row carries a `trace_id` linking back to the trace produced during evaluation.
# MAGIC
# MAGIC ### 3. Versioning with `model_id`
# MAGIC - Setting `model_id="models:/my-agent/1"` groups runs in the MLflow UI so comparisons across iterations are one click away.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 3.3** — switch to a deployed Model Serving endpoint with one line: `mlflow.genai.to_predict_fn(...)`.
# MAGIC