# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 Lab 2.4 — `generate_evals_df` — Synthetic Dataset Generation
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Synthetic Generation** — Use `generate_evals_df` to auto-create representative Q&A pairs from your docs
# MAGIC 2. **Coverage Strategy** — Bias the generator with `agent_description` and `question_guidelines` for edge-case coverage
# MAGIC 3. **Schema Compatibility** — Confirm the output is already in MLflow's eval schema — no remapping needed
# MAGIC 4. **Diversity Audit** — Inspect topic spread across source documents to validate coverage
# MAGIC 5. **End-to-End Eval** — Pipe the synthetic set straight into `mlflow.genai.evaluate()` with the Correctness judge
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 completed
# MAGIC - `databricks-agents` SDK installed (handled in Step 1)
# MAGIC - Foundation Model API access
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Install Packages
# MAGIC
# MAGIC `generate_evals_df` lives in `databricks-agents`. We also pull `mlflow[databricks]>=3.1` and `databricks-openai` so the same notebook can run an end-to-end `evaluate()` after generation.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 STEP 1 - INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" databricks-agents databricks-openai

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Configure the Tutorial Namespace
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 2 - CONFIGURE THE TUTORIAL NAMESPACE
# ============================================================================

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_01"

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
print(f"Namespace:  {CATALOG}.{SCHEMA}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Prepare Documentation Chunks
# MAGIC
# MAGIC `generate_evals_df` accepts a Spark/pandas DataFrame with a `content` column (and optionally a `doc_uri`). For real workloads you would load from a Delta table or UC Volume; we hand-craft 6 Delta Lake chunks for reproducibility.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📄 STEP 3 - PREPARE DOCUMENTATION CHUNKS
# ============================================================================

import pandas as pd

docs = [
    {
        "doc_uri": "delta_lake/overview.md",
        "content": (
            "Delta Lake is an open-source storage layer that provides ACID transactions, "
            "scalable metadata handling, and unified streaming and batch data processing "
            "on top of cloud object stores."
        ),
    },
    {
        "doc_uri": "delta_lake/vacuum.md",
        "content": (
            "VACUUM removes data files no longer referenced by a Delta table that are older "
            "than the retention threshold (default 7 days). Reducing the threshold below "
            "the default can break time travel and concurrent readers."
        ),
    },
    {
        "doc_uri": "delta_lake/zorder.md",
        "content": (
            "Z-ordering is a technique to co-locate related information in the same set of files. "
            "It improves data skipping and therefore query performance for selective filters."
        ),
    },
    {
        "doc_uri": "delta_lake/time_travel.md",
        "content": (
            "Delta Lake supports time travel — querying older snapshots of a table by version "
            "number or timestamp using AS OF clauses. Useful for audits, rollbacks, and reproducible ML."
        ),
    },
    {
        "doc_uri": "delta_lake/schema_evolution.md",
        "content": (
            "Delta Lake enforces schema by default but supports schema evolution when writing "
            "with mergeSchema=true or via ALTER TABLE. Type changes require explicit migration."
        ),
    },
    {
        "doc_uri": "delta_lake/optimize.md",
        "content": (
            "OPTIMIZE compacts small files into larger ones to improve read performance. "
            "Combine with Z-ORDER BY for filter columns to maximise data skipping benefits."
        ),
    },
]

docs_df = spark.createDataFrame(pd.DataFrame(docs))
display(docs_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Generate 20 Synthetic Evals
# MAGIC
# MAGIC `generate_evals_df` uses Databricks-hosted LLMs to:
# MAGIC - Read each chunk
# MAGIC - Produce diverse Q&A pairs covering main topics, paraphrases, and edge cases
# MAGIC - Emit the result in MLflow's evaluation schema (`inputs`, `expectations`, `expected_retrieved_context`) — ready for `evaluate()` with no further mapping
# MAGIC
# MAGIC `agent_description` and `question_guidelines` strongly bias the generator — make them specific.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🤖 STEP 4 - GENERATE 20 SYNTHETIC EVALS
# ============================================================================

from databricks.agents.evals import generate_evals_df

agent_description = (
    "A Q&A assistant for Databricks and Delta Lake. Users ask short, direct questions and "
    "expect concise, factual answers grounded in the documentation provided."
)

question_guidelines = """
- Each question should be answerable from the documentation chunks alone.
- Cover happy-path topics AND edge cases (e.g. what breaks when retention is reduced below 7 days).
- Mix difficulty: half basic definitions, half operational / behavioural questions.
- Phrase questions naturally — as a Databricks engineer would type them.
"""

synthetic_eval_df = generate_evals_df(
    docs=docs_df,
    num_evals=20,
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)

print(f"Generated {synthetic_eval_df.count()} synthetic evals.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Review Diversity
# MAGIC
# MAGIC Eyeball the questions, expected facts, and expected retrieved chunks. Quick checks:
# MAGIC - Are questions paraphrased differently or just slightly reworded?
# MAGIC - Do `expected_facts` actually appear in the cited doc?
# MAGIC - Are edge cases (e.g. retention thresholds, schema evolution failures) represented?
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 5 - REVIEW DIVERSITY
# ============================================================================

display(synthetic_eval_df)


# COMMAND ----------

# ============================================================================
# 🔍 TOPIC SPREAD — GROUP BY SOURCE DOC TO CONFIRM COVERAGE
# ============================================================================

from pyspark.sql import functions as F

# expected_retrieved_context is nested inside the expectations struct
coverage = (
    spark.createDataFrame(synthetic_eval_df)
    .withColumn("doc", F.explode("expectations.expected_retrieved_context.doc_uri"))
    .groupBy("doc")
    .count()
    .orderBy(F.desc("count"))
)
display(coverage)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — (Optional) Persist to UC for Reuse
# MAGIC
# MAGIC Synthetic generation costs tokens — save the result so you don't regenerate it for every notebook run.
# MAGIC

# COMMAND ----------

# ============================================================================
# 💾 STEP 6 - (OPTIONAL) PERSIST TO UC FOR REUSE
# ============================================================================

SYNTHETIC_FQN = f"{CATALOG}.{SCHEMA}.tutorial_eval_synthetic_v1"
spark.createDataFrame(synthetic_eval_df).write.mode("overwrite").saveAsTable(SYNTHETIC_FQN)
print(f"Saved synthetic eval set: {SYNTHETIC_FQN}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Use Directly in `mlflow.genai.evaluate`
# MAGIC
# MAGIC The DataFrame is already in the right schema — no mapping, no renames. We wire up the same `my_agent` shape from Lab 2.3 and run the **Correctness** judge over the synthetic set.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧮 STEP 7 - USE DIRECTLY IN `MLFLOW.GENAI.EVALUATE`
# ============================================================================

from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

def my_agent(messages: list) -> str:
    chat_messages = [
        {"role": "system", "content": "You are a concise Databricks expert. Answer in 2 sentences."},
    ] + [{"role": m["role"], "content": m["content"]} for m in messages]
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=chat_messages,
    )
    return resp.choices[0].message.content

results = mlflow.genai.evaluate(
    data=synthetic_eval_df,
    predict_fn=my_agent,
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
# MAGIC | 6 doc chunks loaded into a DataFrame | ✅ |
# MAGIC | `generate_evals_df` produced 20 Q&A pairs | ✅ |
# MAGIC | Coverage spread across multiple source docs | ✅ |
# MAGIC | Synthetic set persisted as `tutorial_eval_synthetic_v1` | ✅ |
# MAGIC | `evaluate()` ran end-to-end with Correctness judge | ✅ |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Module 2 Outcome — Decision Matrix
# MAGIC
# MAGIC > **Outcome:** *You can create, populate, and version evaluation datasets in Unity Catalog — from hand-curated examples, production traces, and synthetic generation with `generate_evals_df`. You understand when to use each data format.*
# MAGIC
# MAGIC ### When to Use Each Approach
# MAGIC
# MAGIC | Approach | Lab | Best for | Strengths | Watch out for |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | **Hand-curated** | 2.2 | Bootstrapping, regression suites for known-tricky cases, golden ground truth | High control, every row reviewed, comes with `expected_facts` | Coverage limited to what you remembered to write; doesn't scale |
# MAGIC | **From production traces** | 2.3 | Regression on shipped behaviour, real-distribution coverage | Matches actual user queries; updates organically as traffic evolves | Arrives without `expectations` — must annotate before using `Correctness` judge |
# MAGIC | **Synthetic (`generate_evals_df`)** | 2.4 | Coverage breadth, edge cases, cold start when you have docs but no traffic | Scales to hundreds of rows cheaply; covers main topics + edge cases | Quality depends on the generator; needs a human review pass for production use |
# MAGIC
# MAGIC ### Production Recipe — Combine All Three
# MAGIC
# MAGIC Mature eval suites blend the three sources:
# MAGIC
# MAGIC 1. **Hand-curated (~10–30 rows)** — bug-derived regression cases, important customer scenarios, known failure modes.
# MAGIC 2. **Trace-derived (~100–1000 rows)** — sample real prod traffic weekly, annotate, and append. This is your *distribution-realistic* base.
# MAGIC 3. **Synthetic (~50–100 rows)** — fill coverage gaps the first two miss (rare topics, adversarial phrasings, edge cases from new docs).
# MAGIC
# MAGIC All three live as Delta tables in UC, share the same MLflow eval schema, and can be merged into a single dataset for `mlflow.genai.evaluate()`. Versioning comes from Delta — every `merge_records` call creates a new version, so you can pin an eval run to an exact dataset snapshot.
# MAGIC
# MAGIC ### Outcome Coverage Map
# MAGIC
# MAGIC | Outcome Component | Where Covered |
# MAGIC | --- | --- |
# MAGIC | **Create UC datasets** | Lab 2.2 (`create_dataset`), 2.3 (`create_dataset` + traces), 2.4 (`saveAsTable`) |
# MAGIC | **Populate datasets** | Lab 2.2 (`merge_records(records=...)`), 2.3 (`merge_records(traces=...)`), 2.4 (`generate_evals_df`) |
# MAGIC | **Version datasets** | Lab 2.2 Step 6 — `DESCRIBE HISTORY` + `VERSION AS OF` |
# MAGIC | **Three data formats** | Hand-curated (2.2), production-trace (2.3), synthetic (2.4) |
# MAGIC | **When to use each** | Decision matrix above |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Module 2 done.** You now have three ways to build an eval dataset — hand-curated, trace-derived, and synthetic — and you know when to reach for each.
# MAGIC
# MAGIC Next: **Module 3 — Connecting Agents to the Eval Harness** — four `predict_fn` patterns covering local, deployed, registered, and async agents.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. What `generate_evals_df` Does
# MAGIC - Reads document chunks and produces diverse Q&A pairs covering main topics + edge cases.
# MAGIC - Output is a Spark DataFrame already in MLflow's eval schema — `inputs`, `expectations`, `expected_retrieved_context`.
# MAGIC
# MAGIC ### 2. Bias the Generator
# MAGIC - **`agent_description`** anchors tone and audience.
# MAGIC - **`question_guidelines`** is where you push for edge cases, paraphrase variety, and difficulty mix.
# MAGIC
# MAGIC ### 3. Cost vs Coverage
# MAGIC - Use **`num_evals=20`** while iterating, **50–100** for production use.
# MAGIC - Persist the generated set to UC so you don't pay for regeneration on every notebook run.
# MAGIC
# MAGIC ### 4. Module 2 Outcome — When to Use Each Source
# MAGIC - **Outcome:** *You can create, populate, and version evaluation datasets in Unity Catalog — from hand-curated examples, production traces, and synthetic generation with `generate_evals_df`. You understand when to use each data format.*
# MAGIC - **Hand-curated (2.2)** → bootstrap, regression suites, golden ground truth. High control, low scale.
# MAGIC - **Production traces (2.3)** → real-distribution coverage, regression on shipped behavior. No expectations until annotated.
# MAGIC - **Synthetic (2.4)** → coverage breadth, edge cases, cold starts from docs. Needs a human review pass.
# MAGIC - **Production recipe:** combine all three. Hand-curated for known pain points, traces for distribution realism, synthetic to fill gaps.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Module 3 — Connecting Agents to the Eval Harness** — four `predict_fn` patterns covering local, deployed, registered, and async agents.
# MAGIC