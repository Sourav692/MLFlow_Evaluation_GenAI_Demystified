# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2.3 — Build an Evaluation Dataset from Production Traces
# MAGIC
# MAGIC **Goal:** Convert real production traces into an evaluation dataset — the recommended path for enterprise deployments.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Generated 10 traces by calling a small traced agent (simulating prod traffic)
# MAGIC 2. Used `mlflow.search_traces(filter_string=...)` to retrieve recent successful traces
# MAGIC 3. Added those traces to a UC dataset via `eval_dataset.merge_records(traces=...)`
# MAGIC 4. Inspected the auto-extracted `inputs` / `outputs` columns in the dataset UI
# MAGIC 5. Applied the recommended filter: `attributes.status = 'OK'` + a specific trace name
# MAGIC
# MAGIC > **Why this matters:** synthetic Q&A pairs rarely match the messy distribution of real users. Building eval datasets from production traces means you're regression-testing exactly the queries you serve.
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 done; `genai_eval_tutorial.module_01` exists.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure Namespace and MLflow Experiment

# COMMAND ----------

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
# MAGIC ## Step 2 — Build a Tiny Traced Agent
# MAGIC
# MAGIC We need something to *produce* traces. We wrap a Foundation Model API call with `@mlflow.trace` so each invocation lands in the experiment with `inputs` and `outputs` automatically extracted.
# MAGIC
# MAGIC The `name` argument is important — we'll filter on it later when searching traces, so production noise doesn't pollute our eval dataset.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()
mlflow.openai.autolog()

@mlflow.trace(name="my_agent")
def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[
            {"role": "system", "content": "You are a concise Databricks expert. Answer in 2 sentences."},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Simulate Production Traffic (Generate 10 Traces)
# MAGIC
# MAGIC Each call below produces one trace. We use a fixed list of representative questions so the lab is reproducible.

# COMMAND ----------

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
# MAGIC ## Step 4 — Retrieve the Recent, Successful Traces
# MAGIC
# MAGIC The recommended filter for building eval datasets:
# MAGIC - `attributes.status = 'OK'` — drop failed runs
# MAGIC - `attributes.timestamp_ms > <recent>` — keep the window tight
# MAGIC - filter by trace name to scope to *one* agent (critical in shared experiments)

# COMMAND ----------

# Window: from just before this lab started, to now
window_start_ms = run_start_ms - 60_000  # 1-minute safety buffer

traces = mlflow.search_traces(
    experiment_names=[EXPERIMENT_PATH],
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
# MAGIC ## Step 5 — Create the UC Dataset and Merge the Traces
# MAGIC
# MAGIC `merge_records(traces=...)` auto-extracts the agent's input arguments and return value into the `inputs` and `outputs` columns. No hand-mapping required.

# COMMAND ----------

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.create_dataset(name=DATASET_FQN)
eval_dataset = eval_dataset.merge_records(traces=traces)

print(f"Dataset {eval_dataset.name} now has {eval_dataset.to_df().count()} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Inspect the Auto-Extracted Columns
# MAGIC
# MAGIC Notice how `inputs` carries the `question` arg from the function signature, and `outputs` carries the model's reply. This is exactly the schema `mlflow.genai.evaluate()` expects.

# COMMAND ----------

display(eval_dataset.to_df())

# COMMAND ----------

# DBTITLE 1,View as a plain Delta table too
display(spark.table(DATASET_FQN))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — (Optional) Add `expected_facts` Manually for the Correctness Judge
# MAGIC
# MAGIC Datasets built from traces start without ground truth. To enable the **Correctness** judge, a domain expert (or another labelling step) needs to add `expectations`. Below we annotate the first row inline as a demonstration.

# COMMAND ----------

# Pull the first row's request_id so we can re-merge with expectations
first_row = eval_dataset.to_df().limit(1).collect()[0]
request_id = first_row["request_id"] if "request_id" in first_row.asDict() else None

if request_id:
    eval_dataset = eval_dataset.merge_records(records=[{
        "inputs": first_row["inputs"],
        "expectations": {
            "expected_facts": ["data skipping", "co-locates", "query performance"]
        },
    }])
    print("Annotated row 1 with expected_facts.")
else:
    print("Skipping annotation — schema differs from this MLflow version.")

# COMMAND ----------

# MAGIC %md
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
