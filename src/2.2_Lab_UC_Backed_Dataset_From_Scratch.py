# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2.2 — Create a UC-Backed Evaluation Dataset from Scratch
# MAGIC
# MAGIC **Goal:** Build your tutorial's golden dataset — 10 Q&A rows about Databricks / Delta Lake — stored as a Delta table in Unity Catalog.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Created a UC-managed evaluation dataset via `mlflow.genai.datasets.create_dataset(...)`
# MAGIC 2. Added rows with `eval_dataset.merge_records(records=...)`
# MAGIC 3. Included `expected_facts` so the dataset is compatible with the built-in **Correctness** judge
# MAGIC 4. Confirmed the dataset is browseable in Unity Catalog Explorer and queryable via Spark
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 completed — the `genai_eval_tutorial` catalog and `module_01` schema must already exist.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Packages
# MAGIC
# MAGIC `mlflow.genai.datasets` ships with `mlflow[databricks] >= 3.1`. We pin again here so the notebook runs standalone if a learner skipped Module 1.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Point at the Tutorial Namespace
# MAGIC
# MAGIC We reuse the catalog/schema from Lab 1.3. The dataset is materialised as a Delta table under this schema, so the fully qualified name follows the standard `catalog.schema.table` convention.

# COMMAND ----------

CATALOG       = "genai_eval_tutorial"
SCHEMA        = "module_01"
DATASET_TABLE = "tutorial_eval_v1"

UC_SCHEMA   = f"{CATALOG}.{SCHEMA}"
DATASET_FQN = f"{UC_SCHEMA}.{DATASET_TABLE}"

print(f"Will create dataset: {DATASET_FQN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Create the Evaluation Dataset
# MAGIC
# MAGIC `create_dataset` registers a Delta-backed evaluation dataset in UC. The schema follows MLflow's GenAI eval convention — `inputs`, `expectations`, optional `tags` — so it can be passed straight into `mlflow.genai.evaluate()` later.

# COMMAND ----------

import mlflow
import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.create_dataset(name=DATASET_FQN)
print(f"Dataset created: {eval_dataset.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Add 10 Q&A Rows with `expected_facts`
# MAGIC
# MAGIC `expected_facts` is a list of strings the **Correctness** judge will look for in the model's response. The judge passes a row if every fact (or its semantic equivalent) appears — this is far more robust than exact-string matching.

# COMMAND ----------

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
# MAGIC ## Step 5 — Inspect the Dataset
# MAGIC
# MAGIC Two ways to look at the data:
# MAGIC 1. **Catalog Explorer:** left nav → Catalog → `genai_eval_tutorial` → `module_01` → `tutorial_eval_v1`
# MAGIC 2. **Spark SQL** below — the dataset is just a Delta table

# COMMAND ----------

display(spark.table(DATASET_FQN))

# COMMAND ----------

# DBTITLE 1,Confirm the schema matches MLflow eval expectations
spark.sql(f"DESCRIBE TABLE {DATASET_FQN}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Version the Dataset (Delta Time Travel)
# MAGIC
# MAGIC Every `merge_records` call appends a new Delta version. This is how MLflow's eval datasets get *versioning for free* — there is no parallel "dataset registry" service, just Delta. Use `DESCRIBE HISTORY` to inspect versions and `VERSION AS OF` to query an older snapshot.
# MAGIC
# MAGIC **Why versioning matters for evals:**
# MAGIC - Reproduce an old eval run by pinning the dataset version it ran against
# MAGIC - Audit when and why test cases were added or removed
# MAGIC - A/B compare an agent's score on dataset v1 vs v2 to detect coverage drift

# COMMAND ----------

# DBTITLE 1,Inspect dataset history
display(spark.sql(f"DESCRIBE HISTORY {DATASET_FQN}"))

# COMMAND ----------

# DBTITLE 1,Append two more rows — creates a new version
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

# DBTITLE 1,Time-travel back to v0 — original 10 rows
v0 = spark.sql(f"SELECT * FROM {DATASET_FQN} VERSION AS OF 0")
print(f"Version 0 row count: {v0.count()} (expect ~10)")
current = spark.table(DATASET_FQN)
print(f"Current row count:   {current.count()} (expect ~12)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — (Optional) Smoke-Test with `mlflow.genai.evaluate`
# MAGIC
# MAGIC The dataset is already in the right schema — passing it into `evaluate()` requires nothing extra. We run a tiny one-row check so you can confirm the round-trip works before Module 3.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
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
sample = eval_dataset.to_df().limit(2)

results = mlflow.genai.evaluate(
    data=sample,
    predict_fn=lambda question: my_agent(question),
    scorers=[Correctness()],
)

display(results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
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
