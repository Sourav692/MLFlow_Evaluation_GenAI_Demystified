# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2.4 — `generate_evals_df`: Synthetic Dataset Generation
# MAGIC
# MAGIC **Goal:** Use Databricks AI to auto-generate representative eval questions from your documents — covers edge cases you'd miss writing them by hand.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Understood what `generate_evals_df` does — reads your docs, generates diverse Q&A pairs covering main topics + edge cases
# MAGIC 2. Fed it a list of document strings (you can also point it at a Delta table or UC Volume path)
# MAGIC 3. Generated **20 evals** from Delta Lake documentation chunks
# MAGIC 4. Reviewed the diversity of the synthetic set in the notebook UI
# MAGIC 5. Passed the resulting Spark DataFrame straight into `mlflow.genai.evaluate(...)` — it's already in the right schema
# MAGIC
# MAGIC > **Recommendation:** `num_evals=50–100` for production use. We use 20 here to keep iteration cost down.
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 done.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Packages
# MAGIC
# MAGIC `generate_evals_df` lives in `databricks-agents`. We also pull `mlflow[databricks]>=3.1` and `databricks-openai` so the same notebook can run an end-to-end `evaluate()` after generation.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-agents databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configure the Tutorial Namespace

# COMMAND ----------

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
# MAGIC ## Step 3 — Prepare Documentation Chunks
# MAGIC
# MAGIC `generate_evals_df` accepts a Spark/pandas DataFrame with a `content` column (and optionally a `doc_uri`). For real workloads you would load from a Delta table or UC Volume; we hand-craft 6 Delta Lake chunks for reproducibility.

# COMMAND ----------

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
# MAGIC ## Step 4 — Generate 20 Synthetic Evals
# MAGIC
# MAGIC `generate_evals_df` uses Databricks-hosted LLMs to:
# MAGIC - Read each chunk
# MAGIC - Produce diverse Q&A pairs covering main topics, paraphrases, and edge cases
# MAGIC - Emit the result in MLflow's evaluation schema (`inputs`, `expectations`, `expected_retrieved_context`) — ready for `evaluate()` with no further mapping
# MAGIC
# MAGIC `agent_description` and `question_guidelines` strongly bias the generator — make them specific.

# COMMAND ----------

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
# MAGIC ## Step 5 — Review Diversity
# MAGIC
# MAGIC Eyeball the questions, expected facts, and expected retrieved chunks. Quick checks:
# MAGIC - Are questions paraphrased differently or just slightly reworded?
# MAGIC - Do `expected_facts` actually appear in the cited doc?
# MAGIC - Are edge cases (e.g. retention thresholds, schema evolution failures) represented?

# COMMAND ----------

display(synthetic_eval_df)

# COMMAND ----------

# DBTITLE 1,Topic spread — group by source doc to confirm coverage
from pyspark.sql import functions as F

if "expected_retrieved_context" in synthetic_eval_df.columns:
    coverage = (
        synthetic_eval_df
        .withColumn("doc", F.explode("expected_retrieved_context.doc_uri"))
        .groupBy("doc")
        .count()
        .orderBy(F.desc("count"))
    )
    display(coverage)
else:
    print("Schema does not include expected_retrieved_context in this MLflow version.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — (Optional) Persist to UC for Reuse
# MAGIC
# MAGIC Synthetic generation costs tokens — save the result so you don't regenerate it for every notebook run.

# COMMAND ----------

SYNTHETIC_FQN = f"{CATALOG}.{SCHEMA}.tutorial_eval_synthetic_v1"
synthetic_eval_df.write.mode("overwrite").saveAsTable(SYNTHETIC_FQN)
print(f"Saved synthetic eval set: {SYNTHETIC_FQN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Use Directly in `mlflow.genai.evaluate`
# MAGIC
# MAGIC The DataFrame is already in the right schema — no mapping, no renames. We wire up the same `my_agent` shape from Lab 2.3 and run the **Correctness** judge over the synthetic set.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[
            {"role": "system", "content": "You are a concise Databricks expert. Answer in 2 sentences."},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content

results = mlflow.genai.evaluate(
    data=synthetic_eval_df,
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
# MAGIC | 6 doc chunks loaded into a DataFrame | ✅ |
# MAGIC | `generate_evals_df` produced 20 Q&A pairs | ✅ |
# MAGIC | Coverage spread across multiple source docs | ✅ |
# MAGIC | Synthetic set persisted as `tutorial_eval_synthetic_v1` | ✅ |
# MAGIC | `evaluate()` ran end-to-end with Correctness judge | ✅ |
# MAGIC
# MAGIC **Module 2 done.** You now have three ways to build an eval dataset — hand-curated, trace-derived, and synthetic — and you know when to reach for each.
