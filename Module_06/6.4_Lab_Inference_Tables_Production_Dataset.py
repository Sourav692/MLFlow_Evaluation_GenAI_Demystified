# Databricks notebook source
# MAGIC %md
# MAGIC # 📥 Lab 6.4 — Query Inference Tables & Build an Eval Dataset from Production
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Simulated Prod Traffic** — Send 20 plausible production requests to an endpoint with AI Gateway + inference table on
# MAGIC 2. **Spark Query of Logs** — Filter the inference table by recency, status code, and route — surface canonical traffic
# MAGIC 3. **Trace Lookup** — Resolve `databricks_request_id` → MLflow trace via `search_traces` + tag filter
# MAGIC 4. **`merge_records(traces=…)`** — Bootstrap a UC-backed eval dataset directly from production traces — zero reshape code
# MAGIC 5. **Production Quality Baseline** — Run `RelevanceToQuery` + `Safety` + custom `Guidelines` on the production-sourced dataset
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 6.2 (AI Gateway + inference table provisioned)
# MAGIC - Lab 2.2 (`mlflow.genai.datasets` familiarity)
# MAGIC - Lab 4.2 (built-in judges)
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" "databricks-sdk>=0.40"

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Configure Namespace, Endpoint, and Tables
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 1 - CONFIGURE NAMESPACE, ENDPOINT, AND TABLES
# ============================================================================

import mlflow

CATALOG = "main"
SCHEMA  = "genai_eval"
INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.tutorial_agent_payload_request_logs"
PROD_DATASET_FQN = f"{CATALOG}.{SCHEMA}.tutorial_eval_from_prod_v1"
ENDPOINT_NAME = "my-databricks-agent"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"Endpoint        : {ENDPOINT_NAME}")
print(f"Inference table : {INFERENCE_TABLE}")
print(f"Prod dataset    : {PROD_DATASET_FQN}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Generate 20 Simulated Production Requests
# MAGIC
# MAGIC In a real workspace your prod endpoint already gets traffic — skip this step. For the tutorial we synthesise 20 plausible Databricks-themed queries so the inference table has rows to work with. The endpoint must already have AI Gateway + inference table enabled (Lab 6.2).
# MAGIC

# COMMAND ----------

# ============================================================================
# 🤖 STEP 2 - GENERATE 20 SIMULATED PRODUCTION REQUESTS
# ============================================================================

from databricks.sdk import WorkspaceClient
import time, mlflow

w = WorkspaceClient()
oai = w.serving_endpoints.get_open_ai_client()

PROD_QUESTIONS = [
    "What is Delta Lake?",
    "Explain Z-ordering in Delta Lake.",
    "When should I use VACUUM?",
    "How does Unity Catalog handle column-level lineage?",
    "What's the difference between Auto Loader and COPY INTO?",
    "Tell me about Delta Live Tables.",
    "How do I time-travel a Delta table to last week?",
    "What is liquid clustering?",
    "Compare partitioning and Z-ordering.",
    "How does Vector Search index a Delta table?",
    "What endpoints does the Foundation Model API expose?",
    "How do I trace a LangChain agent in MLflow?",
    "Explain Lakeflow Declarative Pipelines.",
    "What is MLflow Model Registry?",
    "How do I log a custom pyfunc model?",
    "What is the default VACUUM retention?",
    "How does Databricks Asset Bundles work?",
    "What is a serving endpoint?",
    "How do I version an evaluation dataset?",
    "What is AI Gateway?",
]

# Tag every request so we can find them later in MLflow trace filters.
mlflow.set_tags({"environment": "prod", "lab": "6.4"})

@mlflow.trace
def call_endpoint(question: str) -> str:
    resp = oai.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content

for q in PROD_QUESTIONS:
    try:
        call_endpoint(q)
    except Exception as e:
        print(f"⚠️  request failed: {e}")
print("✅ Sent 20 simulated requests. Inference table writes can take 1-2 min to land.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Query the Inference Table with Spark
# MAGIC
# MAGIC We pull the last 24 h of traffic, keep only successful (`status_code == 200`) requests, and grab the most recent 50 rows. In production you'd add filters for endpoint version, user cohort, route, etc.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - QUERY THE INFERENCE TABLE WITH SPARK
# ============================================================================

from pyspark.sql import functions as F

inf_table = spark.table(INFERENCE_TABLE)

recent_reqs = (
    inf_table
        .filter(F.col("timestamp_ms") > (F.unix_timestamp() - 86400) * 1000)
        .filter(F.col("status_code") == 200)
        .orderBy("timestamp_ms", ascending=False)
        .limit(50)
)

display(recent_reqs.select("databricks_request_id", "timestamp_ms", "status_code", "execution_duration_ms"))
print(f"{recent_reqs.count()} recent successful requests selected.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Inspect a Raw Request / Response Payload
# MAGIC
# MAGIC Before bootstrapping the dataset, sanity-check that the JSON payloads carry what you expect. The `request` field is the OpenAI-style chat payload; `response` is the model output.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 4 - INSPECT A RAW REQUEST / RESPONSE PAYLOAD
# ============================================================================

display(recent_reqs.select(
    "databricks_request_id",
    F.substring("request", 1, 200).alias("request_preview"),
    F.substring("response", 1, 200).alias("response_preview"),
).limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Look Up MLflow Traces for Those Request IDs
# MAGIC
# MAGIC The inference table tells you *what* was asked. The MLflow trace tells you *how the agent answered* — retrieval spans, tool calls, intermediate state. We bridge them via `databricks_request_id`.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 5 - LOOK UP MLFLOW TRACES FOR THOSE REQUEST IDS
# ============================================================================

req_ids = [r.databricks_request_id for r in recent_reqs.collect()]
print(f"{len(req_ids)} request IDs to look up.")

# Build a SQL IN-list filter for mlflow.search_traces
id_list = ",".join(f"'{r}'" for r in req_ids)
traces = mlflow.search_traces(
    experiment_names=[EXPERIMENT_PATH],
    filter_string=f"tags.`mlflow.databricks.requestId` IN ({id_list})",
)
print(f"Resolved {len(traces)} traces.")
display(traces.head(5) if hasattr(traces, "head") else traces[:5])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Bootstrap an Eval Dataset with `merge_records(traces=…)`
# MAGIC
# MAGIC `merge_records(traces=…)` is the production-traffic equivalent of `merge_records(records=…)` from Lab 2.2 — it auto-extracts `inputs`, `outputs`, and the trace pointer for each row. No reshape code.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 6 - BOOTSTRAP AN EVAL DATASET WITH `MERGE_RECORDS(TRACES=…)`
# ============================================================================

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.create_dataset(
    name=PROD_DATASET_FQN,
    experiment_id=mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id,
)

eval_dataset.merge_records(traces=traces)

display(eval_dataset.to_df().limit(10))
print(f"Production-sourced dataset rows: {eval_dataset.to_df().count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Run Scorers on the Production-Sourced Dataset
# MAGIC
# MAGIC The point of bootstrapping from production isn't just to *have* a dataset — it's to learn what your agent's quality looks like on **real traffic**. Run the same judges from Module 4 against it.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 7 - RUN SCORERS ON THE PRODUCTION-SOURCED DATASET
# ============================================================================

from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, Guidelines

style_rules = Guidelines(
    name="style_rules",
    guidelines=[
        "Response must be in English.",
        "Response must not be longer than 6 sentences.",
        "Response must mention a Databricks feature, product, or concept.",
    ],
)

# Use the deployed endpoint as predict_fn — re-runs of the same prompts, this time scored.
predict_endpoint = mlflow.genai.to_predict_fn(f"endpoints:/{ENDPOINT_NAME}")

# The eval dataset extracted from traces has free-form `inputs`. Reshape to chat schema.
import pandas as pd

prod_df = eval_dataset.to_df().toPandas()

def reshape_row(row):
    inputs = row["inputs"]
    if isinstance(inputs, dict):
        if "messages" in inputs:
            return inputs
        if "question" in inputs:
            return {"messages": [{"role": "user", "content": inputs["question"]}]}
    return {"messages": [{"role": "user", "content": str(inputs)}]}

prod_df["inputs"] = prod_df["inputs"].apply(reshape_row)

results_prod = mlflow.genai.evaluate(
    data=prod_df,
    predict_fn=predict_endpoint,
    scorers=[
        Correctness(),       # only fires on rows with expected_facts; that's OK
        RelevanceToQuery(),
        Safety(),
        style_rules,
    ],
    model_id=f"endpoints:/{ENDPOINT_NAME}",
)

display(results_prod.tables["eval_results"])


# COMMAND ----------

# ============================================================================
# ▶️ PRODUCTION QUALITY BASELINE (AGGREGATE)
# ============================================================================

display(results_prod.tables["eval_results"].selectExpr(
    "AVG(CASE WHEN `relevance_to_query/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS relevance_pass",
    "AVG(CASE WHEN `safety/v1/value`             = 'yes' THEN 1.0 ELSE 0.0 END) AS safety_pass",
    "AVG(CASE WHEN `style_rules/v1/value`        = 'yes' THEN 1.0 ELSE 0.0 END) AS style_pass",
    "COUNT(*) AS rows",
))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Operational Hygiene
# MAGIC
# MAGIC A few habits to lock in:
# MAGIC
# MAGIC | Habit | Why |
# MAGIC | --- | --- |
# MAGIC | **Snapshot weekly** — write a notebook/job that runs Step 5–6 on a schedule | Production drifts. Each snapshot is a versioned dataset (`tutorial_eval_from_prod_vNN`). |
# MAGIC | **Tag the experiment** | `mlflow.set_tag("source", "prod_inference_table")` on the eval run lets you filter dashboards. |
# MAGIC | **Keep the inference table TTL'd** | The audit trail is huge. UC liquid clustering on `timestamp_ms` keeps queries fast. |
# MAGIC | **Strip PII before publishing** | Even with input guardrails, treat the inference table like sensitive data — review rows before pushing them into a shared dataset. |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | 20 simulated production requests sent to the endpoint | ✅ |
# MAGIC | Inference table queried with Spark, filtered to recent successful traffic | ✅ |
# MAGIC | `databricks_request_id` resolved to MLflow traces | ✅ |
# MAGIC | Eval dataset bootstrapped via `merge_records(traces=…)` | ✅ |
# MAGIC | Built-in scorers run on the production-sourced dataset | ✅ |
# MAGIC
# MAGIC Next: **Lab 6.5** — register Safety, Correctness, and a custom Guidelines judge as **continuous monitors** that grade production traces 24/7.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. Why Bootstrap from Production
# MAGIC - Hand-curated datasets miss what's actually being asked.
# MAGIC - Inference table contains the prompts real users send, including the ones nobody on the team thought of.
# MAGIC - `merge_records(traces=…)` makes the conversion mechanical.
# MAGIC
# MAGIC ### 2. Snapshot Discipline
# MAGIC - Run this notebook on a weekly job to versioned dataset names (`tutorial_eval_from_prod_vNN`).
# MAGIC - Tag the eval run with `source=prod_inference_table` for filterable dashboards.
# MAGIC - Strip / redact rows before publishing the dataset broadly.
# MAGIC
# MAGIC ### 3. From Audit to Action
# MAGIC - Inference table answers *what* was asked.
# MAGIC - MLflow trace answers *how the agent answered*.
# MAGIC - `merge_records(traces=…)` joins them into the substrate for Module 6's monitoring loop.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 6.5** — register Safety, Correctness, and a custom Guidelines monitor for continuous production grading.
# MAGIC