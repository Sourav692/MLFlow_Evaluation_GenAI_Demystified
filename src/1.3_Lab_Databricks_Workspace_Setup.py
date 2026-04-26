# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1.3 — Databricks Workspace Setup from Scratch
# MAGIC
# MAGIC **Goal:** Configure your Databricks environment end-to-end so every subsequent lab runs without friction.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Installed `mlflow[databricks]>=3.1` and `databricks-openai` in this notebook
# MAGIC 2. Created a Unity Catalog **catalog + schema** to hold all tutorial assets
# MAGIC 3. Pointed MLflow at a workspace experiment for this tutorial
# MAGIC 4. Verified Foundation Model API access by calling `databricks-claude-opus-4-6`
# MAGIC 5. Captured a sanity-check trace via `mlflow.openai.autolog()` visible in the MLflow UI
# MAGIC
# MAGIC > **Cluster requirement:** DBR 15.4 ML LTS (or newer) on a Serverless or classic compute cluster with internet egress to the Foundation Model API endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The MLflow 3 Evaluation Landscape
# MAGIC
# MAGIC Before we configure anything, here is the full picture of what you'll build over the rest of the tutorial. Every later module slots into one of these layers — keep this map in mind as you go.
# MAGIC
# MAGIC ### Three Evaluation Paradigms (You'll Use All Three)
# MAGIC
# MAGIC 1. **Code-based scorers** — deterministic checks: regex, length bounds, JSON validity, custom Python predicates. Cheap, instant, perfect for hard rules ("must contain a citation", "must be ≤200 words").
# MAGIC 2. **LLM-as-Judge** — a calibrated LLM grades outputs against rubrics or `expected_facts`. Scales semantic quality assessment at a fraction of human cost.
# MAGIC 3. **Human feedback** — domain experts label a sample of outputs. Slow and expensive, but the only ground truth for ambiguous or high-stakes cases. Used to *calibrate* the LLM judges.
# MAGIC
# MAGIC Production stacks **layer all three**: code scorers catch hard violations, LLM judges grade quality at scale, humans validate a sample and correct judge drift.
# MAGIC
# MAGIC ### The Pieces You'll Build (Module → Capability Map)
# MAGIC
# MAGIC | Layer | Module | What it gives you | Key API |
# MAGIC | --- | --- | --- | --- |
# MAGIC | **Setup & Tracing** | 1 *(this module)* | Workspace, experiment, UC namespace, FM API access, traces flowing | `mlflow.openai.autolog()` |
# MAGIC | **Eval datasets** | 2 | Versioned UC-backed datasets from hand/traces/synthetic sources | `mlflow.genai.datasets.create_dataset` |
# MAGIC | **`predict_fn` patterns** | 3 | One `evaluate()` call works against local fns, endpoints, registered models, async | `mlflow.genai.evaluate` |
# MAGIC | **Built-in judges** | 4 | `Correctness`, `RelevanceToQuery`, `RetrievalGroundedness`, `Safety`, `Guidelines` | `mlflow.genai.scorers` |
# MAGIC | **Custom judges** | 5 | Domain-specific scorers tuned to your rubrics and calibrated against humans | `@mlflow.genai.scorer` |
# MAGIC | **Human review** | 6 | Labeling sessions that produce ground truth and improve judges | Review App / labeling sessions |
# MAGIC | **CI/CD gates** | 7 | Block deploys that regress on eval scores | `evaluate()` in pipelines |
# MAGIC | **Production monitoring** | 8 | Continuous scoring on live traffic; alerts on drift | Inference tables + scheduled jobs |
# MAGIC
# MAGIC ### Why MLflow 3 (and the `mlflow.genai` Namespace)
# MAGIC
# MAGIC MLflow 3 unified tracing, evaluation, judges, datasets, and serving artifacts under a single `mlflow.genai` namespace. Earlier versions had fragmented APIs across `mlflow.evaluate`, `mlflow.metrics.genai`, and provider-specific integrations — these still exist but are superseded. **Everything in this tutorial uses the `mlflow.genai` API.**
# MAGIC
# MAGIC ### What This Lab Sets Up (Foundation for All Eight Modules)
# MAGIC
# MAGIC By the end of this notebook, the four runtime pre-requisites every later lab depends on will be in place:
# MAGIC
# MAGIC 1. **Packages** — `mlflow[databricks]>=3.1` + `databricks-openai` installed
# MAGIC 2. **UC namespace** — `genai_eval_tutorial.module_01` catalog/schema + a managed volume for unstructured assets
# MAGIC 3. **Experiment** — `/Users/<you>/genai-eval-tutorial`, the home for every trace and eval run that follows
# MAGIC 4. **Foundation Model API access** — verified call to `databricks-claude-opus-4-6` (the model we'll use as both agent and judge)
# MAGIC 5. **Tracing verified end-to-end** — `mlflow.openai.autolog()` confirmed to write a trace visible in the UI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Packages
# MAGIC
# MAGIC We pin `mlflow[databricks]` to `>=3.1` because GenAI evaluation, tracing, and the new judge framework all require MLflow 3.x. `databricks-openai` gives us a drop-in OpenAI-compatible client that authenticates against your workspace automatically.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configure Unity Catalog
# MAGIC
# MAGIC We create a dedicated catalog and schema so every artifact produced in later labs (eval datasets, judge outputs, traces tables) lives in a predictable location. Replace the defaults below if your workspace requires specific naming.

# COMMAND ----------

# DBTITLE 1,Set tutorial namespace
CATALOG  = "genai_eval_tutorial"
SCHEMA   = "module_01"
VOLUME   = "assets"

print(f"Target namespace: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Create catalog, schema, and a managed volume
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA  {SCHEMA}")

display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Set the MLflow Experiment
# MAGIC
# MAGIC Every trace, evaluation run, and judge call needs an **experiment** to land in. We use a workspace path under your user folder so it shows up in the MLflow UI sidebar.

# COMMAND ----------

import mlflow

# Resolve the current user so the experiment path is unique per learner
USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)

EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"MLflow experiment set to: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Verify Foundation Model API Access
# MAGIC
# MAGIC `DatabricksOpenAI` is an OpenAI-compatible client that uses your workspace credentials automatically — no API key to manage. We use it to call `databricks-claude-opus-4-6`, a Foundation Model API endpoint that ships with every Databricks workspace.
# MAGIC
# MAGIC If this call fails, check:
# MAGIC - Foundation Model APIs are enabled in your workspace
# MAGIC - The cluster has network egress
# MAGIC - You have CAN_QUERY entitlement on the serving endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

resp = client.chat.completions.create(
    model="databricks-claude-opus-4-6",
    messages=[
        {"role": "user", "content": "What is Delta Lake in one sentence?"}
    ],
)

print(resp.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Sanity-Check Trace with `mlflow.openai.autolog()`
# MAGIC
# MAGIC Autologging hooks every OpenAI-compatible call and emits an **MLflow Trace** — the unit of observability you'll inspect, score, and evaluate in every later module. Run the cell, then open the MLflow experiment to confirm the trace appears under the **Traces** tab.

# COMMAND ----------

import mlflow

mlflow.openai.autolog()

resp = client.chat.completions.create(
    model="databricks-claude-opus-4-6",
    messages=[
        {"role": "system", "content": "You are a concise Databricks expert."},
        {"role": "user",   "content": "Name three benefits of MLflow Tracing for GenAI apps."},
    ],
)

print(resp.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify the trace in the UI
# MAGIC
# MAGIC 1. Click the **Experiments** icon in the left nav
# MAGIC 2. Open the experiment at the path printed in Step 3
# MAGIC 3. Switch to the **Traces** tab — you should see one trace with the model name, latency, and token counts
# MAGIC
# MAGIC You can also fetch the most recent trace programmatically:

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
traces = mlflow.search_traces(experiment_ids=[experiment.experiment_id], max_results=1)
display(traces)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete — Module 1 Outcome Coverage
# MAGIC
# MAGIC > **Outcome:** *You understand the full MLflow 3 evaluation landscape and have a working Databricks environment — experiment, UC schema, Foundation Model access — ready for all labs ahead.*
# MAGIC
# MAGIC | Outcome Component | Where It Was Covered | Status |
# MAGIC | --- | --- | --- |
# MAGIC | **MLflow 3 evaluation landscape** | Landscape section above (3 paradigms, module map, why `mlflow.genai`) | ✅ |
# MAGIC | **Working experiment** | Step 3 — `/Users/<you>/genai-eval-tutorial` registered | ✅ |
# MAGIC | **UC schema** | Step 2 — `genai_eval_tutorial.module_01` catalog + schema + volume | ✅ |
# MAGIC | **Foundation Model API access** | Step 4 — verified `databricks-claude-opus-4-6` call | ✅ |
# MAGIC | **Tracing flowing end-to-end** | Step 5 — `mlflow.openai.autolog()` trace visible in UI | ✅ |
# MAGIC | **Packages pinned** | Step 1 — `mlflow[databricks]>=3.1` + `databricks-openai` | ✅ |
# MAGIC
# MAGIC Once every row is green, proceed to **Module 2 — Building Evaluation Datasets**.
