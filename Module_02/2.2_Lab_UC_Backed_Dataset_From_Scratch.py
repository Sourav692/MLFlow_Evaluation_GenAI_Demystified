# Databricks notebook source
# MAGIC %md
# MAGIC # 📚 Lab 2.2 — Create a UC-Backed Evaluation Dataset from Scratch
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **UC-Managed Datasets** — Use `mlflow.genai.datasets.create_dataset(...)` to register a Delta-backed eval set
# MAGIC 2. **Schema Compliance** — Add rows with `merge_records(records=...)` in MLflow's eval schema
# MAGIC 3. **Correctness Judge** — Include `expected_facts` so the built-in Correctness judge can grade outputs
# MAGIC 4. **Versioning via Delta** — Use `DESCRIBE HISTORY` and `VERSION AS OF` to inspect and pin dataset versions
# MAGIC 5. **Round-Trip** — Confirm the dataset feeds straight into `mlflow.genai.evaluate()`
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 completed — `genai_eval_tutorial.module_01` namespace must exist
# MAGIC - `mlflow[databricks]>=3.1` available
# MAGIC - Foundation Model API access for the smoke-test step
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Install Packages
# MAGIC
# MAGIC `mlflow.genai.datasets` ships with `mlflow[databricks] >= 3.1`. We pin again here so the notebook runs standalone if a learner skipped Module 1.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 STEP 1 - INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" databricks-openai

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Point at the Tutorial Namespace
# MAGIC
# MAGIC We reuse the catalog/schema from Lab 1.3. The dataset is materialised as a Delta table under this schema, so the fully qualified name follows the standard `catalog.schema.table` convention.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 2 - POINT AT THE TUTORIAL NAMESPACE
# ============================================================================

CATALOG       = "genai_eval_tutorial"
SCHEMA        = "module_01"
DATASET_TABLE = "tutorial_eval_v1"

UC_SCHEMA   = f"{CATALOG}.{SCHEMA}"
DATASET_FQN = f"{UC_SCHEMA}.{DATASET_TABLE}"

print(f"Will create dataset: {DATASET_FQN}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Create the Evaluation Dataset
# MAGIC
# MAGIC `create_dataset` registers a Delta-backed evaluation dataset in UC. The schema follows MLflow's GenAI eval convention — `inputs`, `expectations`, optional `tags` — so it can be passed straight into `mlflow.genai.evaluate()` later.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 3 - CREATE THE EVALUATION DATASET
# ============================================================================

import mlflow
import mlflow.genai.datasets

try:
    eval_dataset = mlflow.genai.datasets.create_dataset(name=DATASET_FQN)
    print(f"Dataset created: {eval_dataset.name}")
except Exception as e:
    if "TABLE_ALREADY_EXISTS" in str(e):
        eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)
        print(f"Dataset already exists, loaded: {eval_dataset.name}")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Add 10 Q&A Rows with `expected_facts`
# MAGIC
# MAGIC `expected_facts` is a list of strings the **Correctness** judge will look for in the model's response. The judge passes a row if every fact (or its semantic equivalent) appears — this is far more robust than exact-string matching.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📥 STEP 4 - ADD 10 Q&A ROWS WITH `EXPECTED_FACTS`
# ============================================================================

records = [
    {
        "inputs": {"question": "What is Delta Live Tables?"},
        "expectations": {
            "expected_facts": ["declarative pipeline", "streaming", "batch"]
        },
    },
    {
        "inputs": {"question": "How does Unity Catalog handle lineage?"},
        "expectations": {
            "expected_facts": ["column-level lineage", "automated"]
        },
    },
    {
        "inputs": {"question": "What is Z-ordering in Delta Lake?"},
        "expectations": {
            "expected_facts": ["data skipping", "co-locates related data", "improves query performance"]
        },
    },
    {
        "inputs": {"question": "Explain the VACUUM command in Delta Lake."},
        "expectations": {
            "expected_facts": ["removes files no longer referenced", "retention period", "reclaims storage"]
        },
    },
    {
        "inputs": {"question": "What guarantees does Delta Lake provide?"},
        "expectations": {
            "expected_facts": ["ACID transactions", "schema enforcement", "time travel"]
        },
    },
    {
        "inputs": {"question": "What is the Medallion architecture?"},
        "expectations": {
            "expected_facts": ["bronze", "silver", "gold", "layered data refinement"]
        },
    },
    {
        "inputs": {"question": "What is MLflow Tracing?"},
        "expectations": {
            "expected_facts": ["captures inputs and outputs", "spans", "GenAI observability"]
        },
    },
    {
        "inputs": {"question": "How is Photon different from standard Spark execution?"},
        "expectations": {
            "expected_facts": ["vectorized engine", "C++", "compatible with Spark APIs"]
        },
    },
    {
        "inputs": {"question": "What is a Databricks SQL Warehouse?"},
        "expectations": {
            "expected_facts": ["serverless or classic compute", "SQL workloads", "BI tools"]
        },
    },
    {
        "inputs": {"question": "What is the purpose of Auto Loader?"},
        "expectations": {
            "expected_facts": ["incrementally ingests files", "cloud storage", "schema inference"]
        },
    },
]

eval_dataset = eval_dataset.merge_records(records=records)
print(f"Added {len(records)} records to {eval_dataset.name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Inspect the Dataset
# MAGIC
# MAGIC Two ways to look at the data:
# MAGIC 1. **Catalog Explorer:** left nav → Catalog → `genai_eval_tutorial` → `module_01` → `tutorial_eval_v1`
# MAGIC 2. **Spark SQL** below — the dataset is just a Delta table
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 5 - INSPECT THE DATASET
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
# MAGIC ## Step 6 — Version the Dataset (Delta Time Travel)
# MAGIC
# MAGIC Every `merge_records` call appends a new Delta version. This is how MLflow's eval datasets get *versioning for free* — there is no parallel "dataset registry" service, just Delta. Use `DESCRIBE HISTORY` to inspect versions and `VERSION AS OF` to query an older snapshot.
# MAGIC
# MAGIC **Why versioning matters for evals:**
# MAGIC - Reproduce an old eval run by pinning the dataset version it ran against
# MAGIC - Audit when and why test cases were added or removed
# MAGIC - A/B compare an agent's score on dataset v1 vs v2 to detect coverage drift
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 INSPECT DATASET HISTORY
# ============================================================================

display(spark.sql(f"DESCRIBE HISTORY {DATASET_FQN}"))


# COMMAND ----------

# ============================================================================
# 📚 APPEND TWO MORE ROWS — CREATES A NEW VERSION
# ============================================================================

extra_rows = [
    {
        "inputs": {"question": "What is Liquid Clustering in Delta Lake?"},
        "expectations": {"expected_facts": ["incremental clustering", "no manual ZORDER", "improves selective queries"]},
    },
    {
        "inputs": {"question": "How does Predictive Optimization work?"},
        "expectations": {"expected_facts": ["automatic OPTIMIZE", "automatic VACUUM", "managed by Databricks"]},
    },
]
eval_dataset.merge_records(records=extra_rows)
display(spark.sql(f"DESCRIBE HISTORY {DATASET_FQN}").limit(5))


# COMMAND ----------

# ============================================================================
# 📥 TIME-TRAVEL BACK TO V0 — ORIGINAL 10 ROWS
# ============================================================================

v0 = spark.sql(f"SELECT * FROM {DATASET_FQN} VERSION AS OF 0")
print(f"Version 0 row count: {v0.count()} (expect ~10)")
current = spark.table(DATASET_FQN)
print(f"Current row count:   {current.count()} (expect ~12)")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — (Optional) Smoke-Test with `mlflow.genai.evaluate`
# MAGIC
# MAGIC The dataset is already in the right schema — passing it into `evaluate()` requires nothing extra. We run a tiny one-row check so you can confirm the round-trip works before Module 3.
# MAGIC

# COMMAND ----------

# ============================================================================
# ✅ STEP 7 - (OPTIONAL) SMOKE-TEST WITH `MLFLOW.GENAI.EVALUATE`
# ============================================================================

from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
mlflow.set_experiment(f"/Users/{USER_EMAIL}/genai-eval-tutorial")

# Evaluate against just the first 2 rows for the smoke test
sample = eval_dataset.to_df().head(2)

results = mlflow.genai.evaluate(
    data=sample,
    predict_fn=lambda question: my_agent(question),
    scorers=[Correctness()],
)

display(results.tables["eval_results"])


# COMMAND ----------

# DBTITLE 1,Step 8 — Group Traces into Sessions
# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Group Traces into Sessions
# MAGIC
# MAGIC Within the same experiment, you can **aggregate traces into logical sessions** using `mlflow.trace.session` metadata. This is useful for:
# MAGIC - **Multi-turn conversations** — group all turns of a chat into one session
# MAGIC - **A/B testing** — separate traces by configuration or model variant
# MAGIC - **User isolation** — each user's interaction forms its own session
# MAGIC
# MAGIC MLflow stores session metadata on each trace. You can then filter, search, and analyze traces by session in the UI or programmatically with `mlflow.search_traces()`.

# COMMAND ----------

# DBTITLE 1,Step 8 — Group Traces into Sessions
# ============================================================================
# 🔗 STEP 8 - GROUP TRACES INTO SESSIONS
# ============================================================================
# The MLflow Sessions tab groups traces by session_id. Two requirements:
#   1. @mlflow.trace on predict_fn — creates the active trace context
#   2. mlflow.update_current_trace(session_id=...) — sets the session_id field
#      that the UI reads (dedicated parameter, not a generic tag)
# ============================================================================

import uuid
import mlflow
from mlflow.genai.scorers import Correctness
from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
mlflow.set_experiment(f"/Users/{USER_EMAIL}/genai-eval-tutorial")

# --- Session A: Databricks Platform session ---
session_a = f"session-platform-{uuid.uuid4().hex[:8]}"
print(f"▶ Session A: {session_a}")

# --- Session B: Delta Lake deep-dive session ---
session_b = f"session-delta-{uuid.uuid4().hex[:8]}"
print(f"▶ Session B: {session_b}")


# --- Evaluation dataset split into two sessions ---
# Session A questions (platform-focused)
session_a_data = [
    {
        "inputs": {"question": "What is Delta Live Tables?"},
        "expectations": {"expected_facts": ["declarative pipeline", "streaming", "batch"]},
    },
    {
        "inputs": {"question": "How does Unity Catalog handle lineage?"},
        "expectations": {"expected_facts": ["column-level lineage", "automated"]},
    },
    {
        "inputs": {"question": "What is a Databricks SQL Warehouse?"},
        "expectations": {"expected_facts": ["serverless or classic compute", "SQL workloads", "BI tools"]},
    },
]

# Session B questions (Delta Lake focused)
session_b_data = [
    {
        "inputs": {"question": "What is Z-ordering in Delta Lake?"},
        "expectations": {"expected_facts": ["data skipping", "co-locates related data", "improves query performance"]},
    },
    {
        "inputs": {"question": "Explain the VACUUM command in Delta Lake."},
        "expectations": {"expected_facts": ["removes files no longer referenced", "retention period", "reclaims storage"]},
    },
    {
        "inputs": {"question": "What guarantees does Delta Lake provide?"},
        "expectations": {"expected_facts": ["ACID transactions", "schema enforcement", "time travel"]},
    },
]


# --- Define predict functions that set session_id on the trace ---
def make_predict_fn(user_id: str, session_id: str):
    """Create a predict_fn with @mlflow.trace + session_id parameter."""
    @mlflow.trace
    def predict(question: str) -> str:
        # session_id= is the dedicated parameter the MLflow Sessions UI reads
        mlflow.update_current_trace(
            session_id=session_id,
            user=user_id,
        )
        resp = client.chat.completions.create(
            model="databricks-claude-opus-4-6",
            messages=[{"role": "user", "content": question}],
        )
        return resp.choices[0].message.content
    return predict


# --- Run evaluation for Session A ---
print("\n📊 Running evaluation for Session A...")
results_a = mlflow.genai.evaluate(
    data=session_a_data,
    predict_fn=make_predict_fn(user_id="learner-1", session_id=session_a),
    scorers=[Correctness()],
)

# --- Run evaluation for Session B ---
print("\n📊 Running evaluation for Session B...")
results_b = mlflow.genai.evaluate(
    data=session_b_data,
    predict_fn=make_predict_fn(user_id="learner-2", session_id=session_b),
    scorers=[Correctness()],
)

print(f"\n✅ Two sessions created with assessments:")
print(f"   Session A ({session_a}): {len(session_a_data)} traces with Correctness scores")
print(f"   Session B ({session_b}): {len(session_b_data)} traces with Correctness scores")
print("\n🔍 Refresh the Sessions tab in the MLflow UI to see the grouped traces.")

# COMMAND ----------

# DBTITLE 1,Verify Session Grouping
# ============================================================================
# 🔍 VERIFY SESSION GROUPING — Search traces by session
# ============================================================================

import mlflow

experiment = mlflow.get_experiment_by_name(f"/Users/{USER_EMAIL}/genai-eval-tutorial")

# Search traces belonging to Session A (session_id is stored in metadata)
traces_a = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"metadata.`mlflow.trace.session` = '{session_a}'",
)
print(f"Session A traces: {len(traces_a)} (expect 3)")
display(traces_a[["trace_id", "request_time", "state", "execution_duration"]].head())

# Search traces belonging to Session B
traces_b = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"metadata.`mlflow.trace.session` = '{session_b}'",
)
print(f"\nSession B traces: {len(traces_b)} (expect 3)")
display(traces_b[["trace_id", "request_time", "state", "execution_duration"]].head())

print("\n💡 session_id= writes to trace_metadata (not tags).")
print("   The MLflow Sessions tab reads from trace_metadata for grouping.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | UC dataset `tutorial_eval_v1` created | ✅ |
# MAGIC | 10 rows inserted with `expected_facts` | ✅ |
# MAGIC | Dataset visible in Catalog Explorer | ✅ |
# MAGIC | Versioning verified via `DESCRIBE HISTORY` + `VERSION AS OF` | ✅ |
# MAGIC | Smoke-test `evaluate()` returns Correctness scores | ✅ |
# MAGIC
# MAGIC Next: **Lab 2.3** — build a dataset from real production traces.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. Datasets Are Delta Tables
# MAGIC - **`create_dataset`** registers a Delta table in UC — every governance, lineage, and access-control story still applies.
# MAGIC - You can read it with `spark.table(...)` like any other table.
# MAGIC
# MAGIC ### 2. Eval Schema Convention
# MAGIC - **`inputs`** and **`expectations`** are the two columns MLflow's evaluators look for.
# MAGIC - **`expected_facts`** is what the Correctness judge consults — it's a list of substrings/concepts the answer must contain.
# MAGIC
# MAGIC ### 3. Versioning Comes From Delta
# MAGIC - Every `merge_records` call appends a new Delta version — no parallel dataset registry needed.
# MAGIC - **`DESCRIBE HISTORY`** lists versions; **`SELECT ... VERSION AS OF n`** pins an eval run to an exact snapshot.
# MAGIC - Lets you reproduce historical eval runs and detect coverage drift between dataset versions.
# MAGIC
# MAGIC ### 4. Hand-Curation Trade-off
# MAGIC - Hand-written sets give you ground truth control, but coverage is whatever you remember to write.
# MAGIC - Pair this approach with the trace-based (Lab 2.3) and synthetic (Lab 2.4) sets for full coverage.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 2.3** — derive an eval dataset from real production traces (the recommended enterprise path).
# MAGIC