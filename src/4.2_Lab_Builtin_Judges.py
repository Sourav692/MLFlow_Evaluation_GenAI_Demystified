# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4.2 — Built-in Judges: Run All on Tutorial Dataset
# MAGIC
# MAGIC **Goal:** Apply every built-in MLflow judge in a single `evaluate()` call and interpret the results dashboard.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Run **`Correctness`**, **`RelevanceToQuery`**, **`RetrievalGroundedness`**, **`Safety`**, and **`Guidelines`** in one call
# MAGIC 2. Interpreted the results table — per-row scores, aggregate mean, rationale for each judge
# MAGIC 3. Identified the worst-scoring rows and diagnosed root cause from the linked trace
# MAGIC 4. Compared two prompt versions side-by-side using the MLflow Experiments comparison view
# MAGIC
# MAGIC > **Prereq:** Lab 1.3, Lab 2.2 (`tutorial_eval_v1` exists), Lab 3.2 (you understand `predict_fn` semantics).

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure Namespace and Experiment

# COMMAND ----------

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_01"
DATASET_FQN = f"{CATALOG}.{SCHEMA}.tutorial_eval_v1"

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
print(f"Eval data:  {DATASET_FQN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Load the Tutorial Eval Dataset

# COMMAND ----------

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)
print(f"Loaded {eval_dataset.to_df().count()} rows.")
display(eval_dataset.to_df().limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build a RAG-Style Agent
# MAGIC
# MAGIC `RetrievalGroundedness` only fires if the trace contains retrieval spans. We wrap a fake retrieval step (returning hard-coded doc snippets keyed off the question) inside a `@mlflow.trace` so the judge has something to inspect.
# MAGIC
# MAGIC In a real RAG agent, replace `retrieve_docs` with your vector search call.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

DOCS = {
    "delta lake": "Delta Lake is an open-source storage layer that provides ACID transactions, schema enforcement, time travel, and unified streaming/batch on top of cloud object stores.",
    "z-order":    "Z-ordering co-locates related information in the same set of files. It improves data skipping and therefore query performance for selective filters.",
    "vacuum":     "VACUUM removes data files no longer referenced by a Delta table that are older than the retention threshold (default 7 days).",
    "unity catalog": "Unity Catalog provides centralized governance for data and AI on Databricks, including column-level lineage, fine-grained access control, and automated lineage tracking.",
}

@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(question: str) -> list[dict]:
    """Naive keyword match — production would call vector search here."""
    q = question.lower()
    hits = [
        {"page_content": v, "metadata": {"doc_id": k}}
        for k, v in DOCS.items() if k in q
    ]
    return hits or [{"page_content": "No relevant documents found.", "metadata": {"doc_id": "none"}}]

@mlflow.trace
def my_agent(question: str) -> str:
    docs = retrieve_docs(question)
    context = "\n\n".join(d["page_content"] for d in docs)
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": "Answer using only the provided context. Be concise."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content

print(my_agent("What is Z-ordering?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Run All Built-in Judges in One Call
# MAGIC
# MAGIC Each judge answers a different question:
# MAGIC
# MAGIC | Judge | Question Answered | Needs |
# MAGIC | --- | --- | --- |
# MAGIC | **`Correctness`** | Does the answer cover the `expected_facts`? | `expectations.expected_facts` in the dataset |
# MAGIC | **`RelevanceToQuery`** | Does the answer actually address the question? | nothing extra |
# MAGIC | **`RetrievalGroundedness`** | Are claims grounded in the retrieved docs? | a `RETRIEVER` span in the trace |
# MAGIC | **`Safety`** | Is the answer safe (no harmful, biased, or PII content)? | nothing extra |
# MAGIC | **`Guidelines`** | Does the answer follow the listed rules? | a list of natural-language rules |

# COMMAND ----------

from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
    Safety,
    Guidelines,
)

style_guidelines = Guidelines(
    name="style_rules",
    guidelines=[
        "The response must be in English.",
        "The response must not contain marketing language or hype.",
        "The response must be 4 sentences or fewer.",
    ],
)

results_v1 = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        RetrievalGroundedness(),
        Safety(),
        style_guidelines,
    ],
    model_id="models:/my-agent/v1",
)

print("Evaluation v1 complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Interpret the Results Dashboard
# MAGIC
# MAGIC Three things to look at:
# MAGIC 1. **Per-row scores** — every judge produces a `value` ("yes" / "no" / numeric) and a `rationale` column.
# MAGIC 2. **Aggregate metrics** — the run tab shows a pass-rate per judge across the dataset.
# MAGIC 3. **Trace links** — each row has a `trace_id` that opens the full trace (retrieval + LLM call + spans).

# COMMAND ----------

display(results_v1.tables["eval_results"])

# COMMAND ----------

# DBTITLE 1,Aggregate pass rates per judge
agg = results_v1.tables["eval_results"]
display(agg.selectExpr(
    "AVG(CASE WHEN `correctness/v1/value`             = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness_pass",
    "AVG(CASE WHEN `relevance_to_query/v1/value`      = 'yes' THEN 1.0 ELSE 0.0 END) AS relevance_pass",
    "AVG(CASE WHEN `retrieval_groundedness/v1/value`  = 'yes' THEN 1.0 ELSE 0.0 END) AS groundedness_pass",
    "AVG(CASE WHEN `safety/v1/value`                  = 'yes' THEN 1.0 ELSE 0.0 END) AS safety_pass",
    "AVG(CASE WHEN `style_rules/v1/value`             = 'yes' THEN 1.0 ELSE 0.0 END) AS style_pass",
    "COUNT(*) AS rows",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Find Worst-Scoring Rows and Diagnose
# MAGIC
# MAGIC The fastest path from a low score to a fix:
# MAGIC 1. Filter to rows where any judge said "no".
# MAGIC 2. Read the rationale — that tells you which fact / rule failed.
# MAGIC 3. Open the linked trace — read the retrieved context and the LLM call to confirm whether retrieval, prompting, or model is at fault.

# COMMAND ----------

from pyspark.sql import functions as F

failing = (
    results_v1.tables["eval_results"]
    .filter(
        (F.col("correctness/v1/value") != "yes") |
        (F.col("retrieval_groundedness/v1/value") != "yes")
    )
    .select(
        "request_id",
        "inputs",
        "outputs",
        "correctness/v1/value",
        "correctness/v1/rationale",
        "retrieval_groundedness/v1/value",
        "retrieval_groundedness/v1/rationale",
    )
)
display(failing)

# COMMAND ----------

# DBTITLE 1,Open one failing trace by request_id
# Pick the first failing row and search its trace
first = failing.limit(1).collect()
if first:
    rid = first[0]["request_id"]
    trace = mlflow.search_traces(experiment_names=[EXPERIMENT_PATH], filter_string=f"attributes.request_id = '{rid}'")
    display(trace)
else:
    print("No failing rows to diagnose — your agent is doing well!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Compare Two Prompt Versions
# MAGIC
# MAGIC We make a tiny change — instructing the agent to cite doc IDs — then re-evaluate. The MLflow UI's **Compare runs** view (select both runs → Compare) will show side-by-side metric deltas.

# COMMAND ----------

@mlflow.trace
def my_agent_v2(question: str) -> str:
    docs = retrieve_docs(question)
    context = "\n\n".join(f"[{d['metadata']['doc_id']}] {d['page_content']}" for d in docs)
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": (
                "Answer using only the provided context. Be concise. "
                "Cite the doc_id in square brackets after each claim."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content

results_v2 = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent_v2,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        RetrievalGroundedness(),
        Safety(),
        style_guidelines,
    ],
    model_id="models:/my-agent/v2",
)

print("Evaluation v2 complete. Open the experiment UI and compare v1 vs v2.")

# COMMAND ----------

# DBTITLE 1,Side-by-side aggregate comparison
def pass_rates(results, label):
    df = results.tables["eval_results"]
    return df.selectExpr(
        f"'{label}' AS run",
        "AVG(CASE WHEN `correctness/v1/value`            = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness",
        "AVG(CASE WHEN `relevance_to_query/v1/value`     = 'yes' THEN 1.0 ELSE 0.0 END) AS relevance",
        "AVG(CASE WHEN `retrieval_groundedness/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS groundedness",
        "AVG(CASE WHEN `safety/v1/value`                 = 'yes' THEN 1.0 ELSE 0.0 END) AS safety",
        "AVG(CASE WHEN `style_rules/v1/value`            = 'yes' THEN 1.0 ELSE 0.0 END) AS style",
    )

display(pass_rates(results_v1, "v1").union(pass_rates(results_v2, "v2_with_citations")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | All 5 built-in judges ran in a single `evaluate()` call | ✅ |
# MAGIC | Per-row scores + aggregate pass-rates inspected | ✅ |
# MAGIC | Worst-scoring rows isolated and diagnosed via trace | ✅ |
# MAGIC | v1 vs v2 prompt comparison shows judge-level deltas | ✅ |
# MAGIC
# MAGIC Next: **Lab 4.3** — write deterministic code-based scorers with `@scorer` for rules LLM judges can't reliably check.
