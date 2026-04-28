# Databricks notebook source
# MAGIC %md
# MAGIC # ⚖️ Lab 4.2 — Built-in Judges: Run All on Tutorial Dataset
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Five Built-in Judges** — Run `Correctness`, `RelevanceToQuery`, `RetrievalGroundedness`, `Safety`, and `Guidelines` in one call
# MAGIC 2. **Result Interpretation** — Read per-row scores, aggregate pass-rates, and rationale columns
# MAGIC 3. **Failure Diagnosis** — Filter to worst-scoring rows, open the linked trace, identify retrieval vs prompt vs model fault
# MAGIC 4. **RAG Trace Shape** — Wrap a retrieval step in `@mlflow.trace(span_type="RETRIEVER")` so groundedness has spans to read
# MAGIC 5. **Prompt Comparison** — Compare two prompt versions side-by-side using the MLflow Experiments comparison view
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 — workspace setup
# MAGIC - Lab 2.2 — `tutorial_eval_v1` dataset with `expected_facts`
# MAGIC - Lab 3.2 — understand `predict_fn` semantics
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" databricks-openai

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Configure Namespace and Experiment
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 1 - CONFIGURE NAMESPACE AND EXPERIMENT
# ============================================================================

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
# MAGIC ---
# MAGIC ## Step 2 — Load the Tutorial Eval Dataset
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 2 - LOAD THE TUTORIAL EVAL DATASET
# ============================================================================

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)
print(f"Loaded {eval_dataset.to_df().count()} rows.")
display(eval_dataset.to_df().head(3))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Build a RAG-Style Agent
# MAGIC
# MAGIC `RetrievalGroundedness` only fires if the trace contains retrieval spans. We wrap a fake retrieval step (returning hard-coded doc snippets keyed off the question) inside a `@mlflow.trace` so the judge has something to inspect.
# MAGIC
# MAGIC In a real RAG agent, replace `retrieve_docs` with your vector search call.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - BUILD A RAG-STYLE AGENT
# ============================================================================

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
# MAGIC ---
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
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 4 - RUN ALL BUILT-IN JUDGES IN ONE CALL
# ============================================================================

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
    # model_id="models:/my-agent/v1",
)

print("Evaluation v1 complete.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Interpret the Results Dashboard
# MAGIC
# MAGIC Three things to look at:
# MAGIC 1. **Per-row scores** — every judge produces a `value` ("yes" / "no" / numeric) and a `rationale` column.
# MAGIC 2. **Aggregate metrics** — the run tab shows a pass-rate per judge across the dataset.
# MAGIC 3. **Trace links** — each row has a `trace_id` that opens the full trace (retrieval + LLM call + spans).
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 5 - INTERPRET THE RESULTS DASHBOARD
# ============================================================================

display(results_v1.tables["eval_results"])


# COMMAND ----------

# ============================================================================
# ▶️ AGGREGATE PASS RATES PER JUDGE
# ============================================================================

import pandas as pd

agg = results_v1.tables["eval_results"]
summary = pd.DataFrame([{
    "correctness_pass": (agg["correctness/value"] == "yes").mean(),
    "relevance_pass": (agg["relevance_to_query/value"] == "yes").mean(),
    "groundedness_pass": (agg["retrieval_groundedness/value"] == "yes").mean(),
    "safety_pass": (agg["safety/value"] == "yes").mean(),
    "style_pass": (agg["style_rules/value"] == "yes").mean(),
    "rows": len(agg),
}])
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Find Worst-Scoring Rows and Diagnose
# MAGIC
# MAGIC The fastest path from a low score to a fix:
# MAGIC 1. Filter to rows where any judge said "no".
# MAGIC 2. Read the rationale — that tells you which fact / rule failed.
# MAGIC 3. Open the linked trace — read the retrieved context and the LLM call to confirm whether retrieval, prompting, or model is at fault.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📥 STEP 6 - FIND WORST-SCORING ROWS AND DIAGNOSE
# ============================================================================

agg = results_v1.tables["eval_results"]

failing = agg[
    (agg["correctness/value"] != "yes") |
    (agg["retrieval_groundedness/value"] != "yes")
][[
    "trace_id",
    "request",
    "response",
    "correctness/value",
    "retrieval_groundedness/value",
]]
display(failing)

# COMMAND ----------

# ============================================================================
# 🔭 OPEN ONE FAILING TRACE BY REQUEST_ID
# ============================================================================

# Pick the first failing row and search its trace
first = failing.head(1).to_dict('records')
if first:
    rid = first[0]["trace_id"]
    trace = mlflow.get_trace(rid)
    display(trace)
else:
    print("No failing rows to diagnose — your agent is doing well!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Compare Two Prompt Versions
# MAGIC
# MAGIC We make a tiny change — instructing the agent to cite doc IDs — then re-evaluate. The MLflow UI's **Compare runs** view (select both runs → Compare) will show side-by-side metric deltas.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 7 - COMPARE TWO PROMPT VERSIONS
# ============================================================================

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
    ]
)

print("Evaluation v2 complete. Open the experiment UI and compare v1 vs v2.")


# COMMAND ----------

# ============================================================================
# ▶️ SIDE-BY-SIDE AGGREGATE COMPARISON
# ============================================================================

def pass_rates(results, label):
    df = spark.createDataFrame(results.tables["eval_results"])
    return df.selectExpr(
        f"'{label}' AS run",
        "AVG(CASE WHEN `correctness/value`            = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness",
        "AVG(CASE WHEN `relevance_to_query/value`     = 'yes' THEN 1.0 ELSE 0.0 END) AS relevance",
        "AVG(CASE WHEN `retrieval_groundedness/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS groundedness",
        "AVG(CASE WHEN `safety/value`                 = 'yes' THEN 1.0 ELSE 0.0 END) AS safety",
        "AVG(CASE WHEN `style_rules/value`            = 'yes' THEN 1.0 ELSE 0.0 END) AS style",
    )

display(pass_rates(results_v1, "v1").union(pass_rates(results_v2, "v2_with_citations")))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
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
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. What Each Built-in Judge Does
# MAGIC - **`Correctness`** — does the response cover `expected_facts` from the dataset?
# MAGIC - **`RelevanceToQuery`** — does it actually answer the question?
# MAGIC - **`RetrievalGroundedness`** — are claims grounded in retrieved docs? (needs `RETRIEVER` spans)
# MAGIC - **`Safety`** — harmful / biased / PII content detection.
# MAGIC - **`Guidelines`** — pass/fail against a list of natural-language rules.
# MAGIC
# MAGIC ### 2. Diagnosing Failures
# MAGIC - Filter eval results to rows where any judge said `"no"`.
# MAGIC - Read the rationale → that points at the failed fact or rule.
# MAGIC - Open the linked trace → confirm whether retrieval, prompt, or model is at fault.
# MAGIC
# MAGIC ### 3. Comparing Prompt Versions
# MAGIC - Set distinct `model_id` values per run (`models:/my-agent/v1`, `models:/my-agent/v2`).
# MAGIC - Open the experiment, select both runs, click Compare — judge-level deltas show up immediately.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 4.3** — write deterministic `@scorer` code-based scorers for rules LLMs can't reliably check.
# MAGIC