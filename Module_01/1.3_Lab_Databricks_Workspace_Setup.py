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
# MAGIC 4. Verified Foundation Model API access by calling `databricks-claude-sonnet-4`
# MAGIC 5. Captured a sanity-check trace via `mlflow.openai.autolog()` visible in the MLflow UI
# MAGIC
# MAGIC > **Cluster requirement:** DBR 15.4 ML LTS (or newer) on a Serverless or classic compute cluster with internet egress to the Foundation Model API endpoint.

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
# MAGIC `DatabricksOpenAI` is an OpenAI-compatible client that uses your workspace credentials automatically — no API key to manage. We use it to call `databricks-claude-sonnet-4`, a Foundation Model API endpoint that ships with every Databricks workspace.
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
    model="databricks-claude-sonnet-4",
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
    model="databricks-claude-sonnet-4",
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

traces = mlflow.search_traces(experiment_names=[EXPERIMENT_PATH], max_results=1)
display(traces)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | `mlflow[databricks]>=3.1` installed | ✅ |
# MAGIC | UC catalog + schema + volume created | ✅ |
# MAGIC | MLflow experiment registered under your user folder | ✅ |
# MAGIC | Foundation Model API call succeeded | ✅ |
# MAGIC | Sanity-check trace visible in the MLflow UI | ✅ |
# MAGIC
# MAGIC Proceed to **Module 2** — `Tracing Fundamentals` — once every row is green.
