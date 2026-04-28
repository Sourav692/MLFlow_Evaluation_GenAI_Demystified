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

eval_dataset = mlflow.genai.datasets.create_dataset(name=DATASET_FQN)
print(f"Dataset created: {eval_dataset.name}")


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