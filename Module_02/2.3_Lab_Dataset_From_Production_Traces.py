# Databricks notebook source
# MAGIC %md
# MAGIC # 🔭 Lab 2.3 — Build an Evaluation Dataset from Production Traces
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Traced Agent** — Wrap an LLM call with `@mlflow.trace` so every invocation lands in your experiment
# MAGIC 2. **Trace Search** — Use `mlflow.search_traces(filter_string=...)` to retrieve recent successful runs
# MAGIC 3. **Trace → Dataset** — Add traces to a UC dataset via `merge_records(traces=...)` — inputs/outputs auto-extract
# MAGIC 4. **Production Filtering** — Apply `attributes.status = 'OK'` and `attributes.name = ...` to scope the eval set cleanly
# MAGIC 5. **Annotation Pattern** — Add `expected_facts` after the fact so the Correctness judge can grade trace-derived rows
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 completed — namespace and experiment exist
# MAGIC - Foundation Model API endpoint reachable from the cluster
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
# MAGIC ## Step 1 — Configure Namespace and MLflow Experiment
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 1 - CONFIGURE NAMESPACE AND MLFLOW EXPERIMENT
# ============================================================================

import mlflow

CATALOG       = "genai_eval_tutorial"
SCHEMA        = "module_01"
DATASET_TABLE = "tutorial_eval_from_traces_v1"

UC_SCHEMA   = f"{CATALOG}.{SCHEMA}"
DATASET_FQN = f"{UC_SCHEMA}.{DATASET_TABLE}"

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
print(f"Target dataset: {DATASET_FQN}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Build a Tiny Traced Agent
# MAGIC
# MAGIC We need something to *produce* traces. We wrap a Foundation Model API call with `@mlflow.trace` so each invocation lands in the experiment with `inputs` and `outputs` automatically extracted.
# MAGIC
# MAGIC The `name` argument is important — we'll filter on it later when searching traces, so production noise doesn't pollute our eval dataset.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 2 - BUILD A TINY TRACED AGENT
# ============================================================================

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()
mlflow.openai.autolog()

@mlflow.trace(name="my_agent")
def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": "You are a concise Databricks expert. Answer in 2 sentences."},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Simulate Production Traffic (Generate 10 Traces)
# MAGIC
# MAGIC Each call below produces one trace. We use a fixed list of representative questions so the lab is reproducible.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🤖 STEP 3 - SIMULATE PRODUCTION TRAFFIC (GENERATE 10 TRACES)
# ============================================================================

import time

questions = [
    "What is Z-ordering in Delta Lake?",
    "Explain the VACUUM command.",
    "What does Auto Loader do?",
    "How does Unity Catalog secure tables?",
    "What is Photon?",
    "When should I use Delta Live Tables vs plain Delta?",
    "What is the medallion architecture?",
    "How does time travel work in Delta?",
    "What is MLflow Tracing for?",
    "Explain Databricks SQL Serverless.",
]

# Capture the start timestamp BEFORE the agent runs so we don't pull in older traces
run_start_ms = int(time.time() * 1000)

for q in questions:
    print(my_agent(q)[:80] + "...")

print(f"\nGenerated {len(questions)} traces.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Retrieve the Recent, Successful Traces
# MAGIC
# MAGIC The recommended filter for building eval datasets:
# MAGIC - `attributes.status = 'OK'` — drop failed runs
# MAGIC - `attributes.timestamp_ms > <recent>` — keep the window tight
# MAGIC - filter by trace name to scope to *one* agent (critical in shared experiments)
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 4 - RETRIEVE THE RECENT, SUCCESSFUL TRACES
# ============================================================================

# Window: from just before this lab started, to now
window_start_ms = run_start_ms - 60_000  # 1-minute safety buffer

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
# traces = mlflow.search_traces(experiment_ids=[experiment.experiment_id], max_results=1)

traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string=(
        f"attributes.timestamp_ms > {window_start_ms}"
        f" AND attributes.status = 'OK'"
        f" AND attributes.name = 'my_agent'"
    ),
    order_by=["attributes.timestamp_ms DESC"],
)

print(f"Retrieved {len(traces)} traces.")
display(traces)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Create the UC Dataset and Merge the Traces
# MAGIC
# MAGIC `merge_records(traces=...)` auto-extracts the agent's input arguments and return value into the `inputs` and `outputs` columns. No hand-mapping required.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 5 - CREATE THE UC DATASET AND MERGE THE TRACES
# ============================================================================

import mlflow.genai.datasets

try:
    eval_dataset = mlflow.genai.datasets.create_dataset(name=DATASET_FQN)
except Exception:
    eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)

# Convert traces to records with inputs/outputs columns
records = [
    {"inputs": row["request"], "outputs": {"output": row["response"]}}
    for _, row in traces.iterrows()
]
eval_dataset = eval_dataset.merge_records(records)

print(f"Dataset {eval_dataset.name} now has {eval_dataset.to_df().count()} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Inspect the Auto-Extracted Columns
# MAGIC
# MAGIC Notice how `inputs` carries the `question` arg from the function signature, and `outputs` carries the model's reply. This is exactly the schema `mlflow.genai.evaluate()` expects.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 6 - INSPECT THE AUTO-EXTRACTED COLUMNS
# ============================================================================

display(eval_dataset.to_df())


# COMMAND ----------

# ============================================================================
# ▶️ VIEW AS A PLAIN DELTA TABLE TOO
# ============================================================================

display(spark.table(DATASET_FQN))


# COMMAND ----------

# ============================================================================
# 🗂️ CONFIRM THE SCHEMA MATCHES MLFLOW EVAL EXPECTATIONS
# ============================================================================

spark.sql(f"DESCRIBE TABLE {DATASET_FQN}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — (Optional) Add `expected_facts` Manually for the Correctness Judge
# MAGIC
# MAGIC Datasets built from traces start without ground truth. To enable the **Correctness** judge, a domain expert (or another labelling step) needs to add `expectations`. Below we annotate the first row inline as a demonstration.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧩 STEP 7 - (OPTIONAL) ADD `EXPECTED_FACTS` MANUALLY FOR THE CORRECTNESS JUDGE
# ============================================================================

# Pull the first row so we can re-merge with expectations
first_row = eval_dataset.to_df().head(1).iloc[0]
record_id = first_row.get("dataset_record_id") or first_row.get("request_id")

if record_id:
    eval_dataset = eval_dataset.merge_records(records=[{
        "inputs": first_row["inputs"],
        "expectations": '{"expected_facts": ["data skipping", "co-locates", "query performance"]}',
    }])
    print("Annotated row 1 with expected_facts.")
else:
    print("Skipping annotation — schema differs from this MLflow version.")

# COMMAND ----------

# ============================================================================
# ▶️ VIEW AS A PLAIN DELTA TABLE TOO
# ============================================================================

display(spark.table(DATASET_FQN))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | 10 traces generated under the `my_agent` name | ✅ |
# MAGIC | `search_traces` returned only recent + OK + `my_agent` traces | ✅ |
# MAGIC | UC dataset `tutorial_eval_from_traces_v1` populated | ✅ |
# MAGIC | `inputs` and `outputs` auto-extracted from traces | ✅ |
# MAGIC
# MAGIC Next: **Lab 2.4** — let Databricks AI synthesise a diverse eval set straight from your documents.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. Why Build From Traces
# MAGIC - **Real distribution.** Synthetic Q&A rarely matches the messy queries you actually serve.
# MAGIC - **Regression coverage.** Each trace becomes a regression test for behavior you've already shipped.
# MAGIC
# MAGIC ### 2. Filter Hygiene
# MAGIC - **`status = 'OK'`** drops failed runs that would poison the eval set.
# MAGIC - **`name = 'my_agent'`** scopes to one agent in shared experiments.
# MAGIC - **`timestamp_ms > <recent>`** keeps the window tight — capture the cut-off *before* the agent runs.
# MAGIC
# MAGIC ### 3. Auto-Extraction
# MAGIC - **`merge_records(traces=...)`** maps function arguments → `inputs` and return value → `outputs` with no hand-mapping.
# MAGIC - Trace-derived rows arrive without `expectations` — you must add ground truth before the Correctness judge will work.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 2.4** — let Databricks AI synthesize a diverse eval set straight from your documents.
# MAGIC