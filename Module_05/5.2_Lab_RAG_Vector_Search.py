# Databricks notebook source
# MAGIC %md
# MAGIC # 🔎 Lab 5.2 — Build & Evaluate a RAG Pipeline on Databricks Vector Search
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Vector Search Index** — Build a Delta-synced Vector Search index over Databricks doc chunks in Unity Catalog
# MAGIC 2. **Traced RAG Agent** — Two-span agent — RETRIEVER + LLM — both inspectable in the MLflow trace UI
# MAGIC 3. **Three-Pillar Eval** — Run `RetrievalGroundedness` + `Correctness` + `retrieved_document_recall` in one `evaluate()`
# MAGIC 4. **Custom Recall Scorer** — Trace-based `@scorer` that reads RETRIEVER spans and computes recall@K against expected doc_ids
# MAGIC 5. **Failure-Mode Diagnosis** — Distinguish retrieval miss vs. hallucination from the score grid
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 (workspace + UC + Foundation Model API)
# MAGIC - Lab 2.2 (`tutorial_eval_v1` dataset exists)
# MAGIC - Lab 4.2 + 4.3 (built-in judges + `@scorer` patterns)
# MAGIC - Vector Search enabled in your workspace
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" databricks-openai databricks-vectorsearch

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Configure Namespace, Experiment, and Tracing
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗂️ STEP 1 - CONFIGURE NAMESPACE, EXPERIMENT, AND TRACING
# ============================================================================

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_05"
DOCS_TABLE   = f"{CATALOG}.{SCHEMA}.docs_chunks"
VS_ENDPOINT  = "genai_eval_tutorial_vs"
VS_INDEX_FQN = f"{CATALOG}.{SCHEMA}.docs_chunks_index"
EVAL_DATASET_FQN = f"{CATALOG}.module_01.tutorial_eval_v1"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

# Autolog: every OpenAI-compatible call (retrieval-side embeddings + LLM generation) is auto-traced.
mlflow.openai.autolog()

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"Experiment : {EXPERIMENT_PATH}")
print(f"Docs table : {DOCS_TABLE}")
print(f"VS index   : {VS_INDEX_FQN}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Create the Source Delta Table of Doc Chunks
# MAGIC
# MAGIC Vector Search builds an index **on top of a Delta table**. Each row is one retrievable chunk.
# MAGIC
# MAGIC > In production, chunks come from a real document ingestion pipeline (PDFs split into ~500-token windows). Here we hand-author 8 short chunks covering the same Databricks topics our eval dataset asks about — small enough to inspect end-to-end.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 2 - CREATE THE SOURCE DELTA TABLE OF DOC CHUNKS
# ============================================================================

from pyspark.sql import Row

chunks = [
    Row(doc_id="delta_lake",     content="Delta Lake is an open-source storage layer that provides ACID transactions, schema enforcement, time travel, and unified streaming and batch processing on top of cloud object stores."),
    Row(doc_id="z_order",        content="Z-ordering co-locates related information in the same set of files. It improves data skipping and therefore query performance for selective filters on the Z-ordered columns."),
    Row(doc_id="vacuum",         content="VACUUM removes data files no longer referenced by a Delta table that are older than the retention threshold. The default retention is 7 days."),
    Row(doc_id="time_travel",    content="Delta Lake time travel lets you query a previous snapshot of a table using VERSION AS OF or TIMESTAMP AS OF. It is bounded by the VACUUM retention window."),
    Row(doc_id="unity_catalog",  content="Unity Catalog provides centralized governance for data and AI on Databricks, including column-level lineage, fine-grained access control, and automated lineage tracking."),
    Row(doc_id="liquid_cluster", content="Liquid clustering is an alternative to partitioning and Z-ordering on Delta. It adapts cluster keys without rewriting data and works well when access patterns evolve."),
    Row(doc_id="dlt_pipelines",  content="Delta Live Tables (Lakeflow Declarative Pipelines) is a declarative framework for building reliable streaming and batch ETL with automatic dependency tracking and data quality expectations."),
    Row(doc_id="autoloader",     content="Auto Loader incrementally and efficiently processes new data files arriving in cloud storage. It tracks file state and supports schema inference and evolution."),
]

(spark.createDataFrame(chunks)
       .write.mode("overwrite")
       .option("delta.enableChangeDataFeed", "true")
       .saveAsTable(DOCS_TABLE))

display(spark.table(DOCS_TABLE))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Create the Vector Search Endpoint and Index
# MAGIC
# MAGIC We use a **Databricks-managed embedding** (`databricks-gte-large-en`) so we don't have to compute embeddings ourselves. Vector Search reads the source Delta table, embeds the `content` column, and keeps the index in sync via Change Data Feed.
# MAGIC
# MAGIC > Endpoint creation is **idempotent** but slow on first run (a few minutes). The cell below skips creation if the endpoint already exists.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 3 - CREATE THE VECTOR SEARCH ENDPOINT AND INDEX
# ============================================================================

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"Creating endpoint {VS_ENDPOINT} (this may take a few minutes)…")
else:
    print(f"Endpoint {VS_ENDPOINT} already exists.")

vsc.wait_for_endpoint(VS_ENDPOINT, timeout=1200)
print(f"Endpoint {VS_ENDPOINT} is ONLINE.")


# COMMAND ----------

# ============================================================================
# ▶️ STEP
# ============================================================================

try:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_FQN)
    print(f"Index {VS_INDEX_FQN} already exists.")
except Exception:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX_FQN,
        source_table_name=DOCS_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="doc_id",
        embedding_source_column="content",
        embedding_model_endpoint_name="databricks-gte-large-en",
    )
    print(f"Created index {VS_INDEX_FQN}.")

index.wait_until_ready(verbose=True, timeout=600)
print("Index is READY.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Build the Traced RAG Agent
# MAGIC
# MAGIC Two spans, both meaningful for evaluation:
# MAGIC
# MAGIC | Span | Why the judges need it |
# MAGIC | --- | --- |
# MAGIC | `RETRIEVER` span (`retrieve()`) | `RetrievalGroundedness` reads the retrieved docs to check the answer is grounded; `retrieved_document_recall` counts which expected doc_ids made it into the top-K |
# MAGIC | LLM span (`mlflow.openai.autolog()`) | `Correctness` reads the final output |
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 4 - BUILD THE TRACED RAG AGENT
# ============================================================================

from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType

llm_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
TOP_K = 3

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str) -> list[dict]:
    """Vector-search retrieval. Returns top-K chunks shaped for RetrievalGroundedness."""
    res = index.similarity_search(
        query_text=question,
        columns=["doc_id", "content"],
        num_results=TOP_K,
    )
    rows = res.get("result", {}).get("data_array", [])
    # data_array rows look like [doc_id, content, score]
    return [
        {"page_content": r[1], "metadata": {"doc_id": r[0], "score": r[2]}}
        for r in rows
    ]

@mlflow.trace
def rag_agent(question: str) -> str:
    docs = retrieve(question)
    context = "\n".join(f"[{d['metadata']['doc_id']}] {d['page_content']}" for d in docs)
    resp = llm_client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": (
                "Answer using only the provided context. "
                "Cite the doc_id in square brackets after each claim. Be concise."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content

print(rag_agent("What is Z-ordering?"))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Custom Retrieval Recall Scorer
# MAGIC
# MAGIC Built-in judges measure **groundedness** (was the answer based on retrieved text?) but not **recall** (did retrieval surface the right docs in the first place?). We add a deterministic code scorer for that — exactly the Module 4 Pattern 3 (trace-based) shape.
# MAGIC
# MAGIC > For this to work, the eval dataset row needs an `expectations.expected_doc_ids` field. Lab 2.2's hand-curated dataset doesn't include it, so we infer expected doc_ids from each question via a tiny keyword map. In a real RAG eval, label this column once and persist it.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 5 - CUSTOM RETRIEVAL RECALL SCORER
# ============================================================================

import mlflow.genai.datasets
from mlflow.entities import Feedback, AssessmentSource, SpanType
from mlflow.genai.scorers import scorer

EXPECTED_DOCS_BY_KEYWORD = {
    "delta lake":     ["delta_lake"],
    "z-order":        ["z_order"],
    "z order":        ["z_order"],
    "vacuum":         ["vacuum"],
    "time travel":    ["time_travel", "delta_lake"],
    "unity catalog":  ["unity_catalog"],
    "liquid":         ["liquid_cluster"],
    "delta live":     ["dlt_pipelines"],
    "lakeflow":       ["dlt_pipelines"],
    "auto loader":    ["autoloader"],
    "autoloader":     ["autoloader"],
}

def expected_doc_ids_for(question: str) -> list[str]:
    q = question.lower()
    hits = []
    for kw, ids in EXPECTED_DOCS_BY_KEYWORD.items():
        if kw in q:
            hits.extend(ids)
    return list(dict.fromkeys(hits))  # dedupe, preserve order

@scorer
def retrieved_document_recall(inputs, trace) -> Feedback:
    """Pattern 3 (trace-based): recall@K = |expected ∩ retrieved| / |expected|."""
    question = inputs.get("question") if isinstance(inputs, dict) else str(inputs)
    expected = set(expected_doc_ids_for(question or ""))
    if not expected:
        return Feedback(
            value=None,
            rationale="No expected_doc_ids known for this question — skipping.",
            source=AssessmentSource(source_type="CODE", source_id="retrieval_recall_v1"),
        )
    retr_spans = trace.search_spans(span_type=SpanType.RETRIEVER)
    retrieved = set()
    for span in retr_spans:
        for d in (span.outputs or []):
            doc_id = (d.get("metadata") or {}).get("doc_id")
            if doc_id:
                retrieved.add(doc_id)
    recall = len(expected & retrieved) / len(expected)
    return Feedback(
        value=round(recall, 2),
        rationale=f"expected={sorted(expected)} retrieved={sorted(retrieved)} recall={recall:.2f}",
        source=AssessmentSource(source_type="CODE", source_id="retrieval_recall_v1"),
    )

print("Custom retrieval recall scorer ready.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Run All Three Pillars in One `evaluate()`
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧮 STEP 6 - RUN ALL THREE PILLARS IN ONE `EVALUATE()`
# ============================================================================

from mlflow.genai.scorers import Correctness, RetrievalGroundedness

eval_dataset = mlflow.genai.datasets.get_dataset(name=EVAL_DATASET_FQN)

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=rag_agent,
    scorers=[
        Correctness(),                # generation pillar (matches expected_facts)
        RetrievalGroundedness(),      # generation grounded in retrieval
        retrieved_document_recall,    # retrieval pillar (recall@K)
    ],
    model_id="models:/rag-agent/v1",
)

display(results.tables["eval_results"])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Diagnose Failure Modes
# MAGIC
# MAGIC The three columns together form a **diagnostic grid**:
# MAGIC
# MAGIC | Correctness | Groundedness | Recall | Diagnosis | Fix |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | ✗ | ✓ | low | **Retrieval miss** — the right doc never made it to context | Improve chunking / embeddings / top-K |
# MAGIC | ✗ | ✗ | high | **Hallucination** — right docs retrieved, model invented | Tighter prompt, smaller temperature, smarter model |
# MAGIC | ✗ | ✓ | high | **Incomplete answer** — context was right but answer skipped facts | Prompt for completeness, verify token budget |
# MAGIC | ✓ | ✓ | low | **Lucky** — model knew without retrieval (don't celebrate) | Add explicit grounding requirement |
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ RETRIEVAL MISSES (GROUNDEDNESS OK, CORRECTNESS FAILS, LOW RECALL)
# ============================================================================

display(results.tables["eval_results"].selectExpr(
    "request_id", "inputs", "outputs",
    "`correctness/v1/value` AS correctness",
    "`retrieval_groundedness/v1/value` AS groundedness",
    "`retrieved_document_recall/v1/value` AS recall",
    "`retrieved_document_recall/v1/rationale` AS recall_detail",
).filter(
    "`correctness/v1/value` != 'yes' "
    "AND `retrieval_groundedness/v1/value` = 'yes' "
    "AND `retrieved_document_recall/v1/value` < 0.5"
))


# COMMAND ----------

# ============================================================================
# ▶️ HALLUCINATIONS (HIGH RECALL, LOW GROUNDEDNESS)
# ============================================================================

display(results.tables["eval_results"].selectExpr(
    "request_id", "inputs", "outputs",
    "`correctness/v1/value` AS correctness",
    "`retrieval_groundedness/v1/value` AS groundedness",
    "`retrieval_groundedness/v1/rationale` AS groundedness_detail",
    "`retrieved_document_recall/v1/value` AS recall",
).filter(
    "`retrieval_groundedness/v1/value` != 'yes' "
    "AND `retrieved_document_recall/v1/value` >= 0.5"
))


# COMMAND ----------

# ============================================================================
# ▶️ AGGREGATE THREE-PILLAR PASS RATES
# ============================================================================

display(results.tables["eval_results"].selectExpr(
    "AVG(CASE WHEN `correctness/v1/value`            = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness_rate",
    "AVG(CASE WHEN `retrieval_groundedness/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS groundedness_rate",
    "AVG(`retrieved_document_recall/v1/value`)                                       AS mean_recall",
    "COUNT(*) AS rows",
))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | Vector Search index built on Delta doc-chunks table | ✅ |
# MAGIC | RAG agent traced — RETRIEVER span + LLM span both visible | ✅ |
# MAGIC | `mlflow.openai.autolog()` enabled for generation tracing | ✅ |
# MAGIC | `RetrievalGroundedness` + `Correctness` + `retrieved_document_recall` all running | ✅ |
# MAGIC | Failure modes split into retrieval-miss vs. hallucination cohorts | ✅ |
# MAGIC
# MAGIC Next: **Lab 5.4** — agent-specific judges (`ToolCallEfficiency`, `KnowledgeRetention`) for tool-using and multi-turn agents.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. The Three RAG Quality Pillars
# MAGIC - **Retrieval recall** — did the right doc make it into the top-K?
# MAGIC - **Groundedness** — is the answer based on the retrieved context, or invented?
# MAGIC - **Correctness** — does the final answer cover the expected facts?
# MAGIC
# MAGIC ### 2. Failure-Mode Grid
# MAGIC - Correctness=✗, Groundedness=✓, Recall=low → **retrieval miss** (fix chunking / embeddings / top-K).
# MAGIC - Correctness=✗, Groundedness=✗, Recall=high → **hallucination** (tighter prompt, smaller temperature).
# MAGIC - Correctness=✗, Groundedness=✓, Recall=high → **incomplete answer** (prompt for completeness).
# MAGIC - Correctness=✓, Groundedness=✗, Recall=low → model knew it without retrieval — don't celebrate.
# MAGIC
# MAGIC ### 3. Why Tracing Is Non-Negotiable
# MAGIC - Both judges and code scorers read from spans. No traces → no diagnosis.
# MAGIC - `mlflow.openai.autolog()` plus `@mlflow.trace(span_type=SpanType.RETRIEVER)` gives full visibility for free.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 5.4** — agent-specific judges (`ToolCallEfficiency`, `KnowledgeRetention`) for tool-using and multi-turn agents.
# MAGIC