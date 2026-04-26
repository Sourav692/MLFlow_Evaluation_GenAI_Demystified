# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 5.2b — RAG on Databricks Vector Search **with LangGraph**
# MAGIC
# MAGIC **Goal:** Same three-pillar RAG evaluation as Lab 5.2, but the agent is built as a **LangGraph `StateGraph`** instead of a hand-written OpenAI loop. Demonstrates that the MLflow eval harness is framework-agnostic — judges and code scorers don't care whether the agent is raw Python, LangChain, or LangGraph.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Built a **LangGraph `StateGraph`** with explicit `retrieve` and `generate` nodes
# MAGIC 2. Used **`DatabricksVectorSearch`** as a LangChain retriever (against the index built in Lab 5.2)
# MAGIC 3. Used **`ChatDatabricks`** as the LLM, with **`mlflow.langchain.autolog()`** for tracing
# MAGIC 4. Confirmed the trace contains a `RETRIEVER` span (so `RetrievalGroundedness` works) and an `LLM` span
# MAGIC 5. Run **`RetrievalGroundedness` + `Correctness` + `retrieved_document_recall`** on the LangGraph agent — identical scorers to Lab 5.2
# MAGIC 6. Compared per-row scores between the **raw-OpenAI agent (Lab 5.2)** and the **LangGraph agent (this lab)** to validate framework parity
# MAGIC
# MAGIC > **Why this lab matters:** The eval contract is `predict_fn(question) -> str` plus a trace with the right span types. *Anything* satisfying that contract can be evaluated. LangGraph just gives you a clearer mental model — nodes for retrieve/generate, edges for control flow — when graphs grow beyond two steps.
# MAGIC
# MAGIC > **Prereq:** Lab 5.2 (Vector Search index `genai_eval_tutorial.module_05.docs_chunks_index` already exists), Lab 4.2/4.3 (judges + `@scorer`).

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" databricks-vectorsearch databricks-langchain langgraph langchain-core
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure Namespace, Experiment, and LangChain Autolog

# COMMAND ----------

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_05"
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

# LangChain autolog turns every LangChain/LangGraph node into an MLflow span automatically.
# Retriever nodes are tagged with span_type=RETRIEVER, LLM nodes with span_type=LLM, etc.
mlflow.langchain.autolog()

print(f"Experiment   : {EXPERIMENT_PATH}")
print(f"Vector index : {VS_INDEX_FQN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Wire Databricks-Native LangChain Components
# MAGIC
# MAGIC | Component | What it does | Why it matters for eval |
# MAGIC | --- | --- | --- |
# MAGIC | `DatabricksVectorSearch` | LangChain retriever wrapping the Vector Search index | Auto-emits `RETRIEVER` spans → `RetrievalGroundedness` works |
# MAGIC | `ChatDatabricks` | LangChain chat model on Foundation Model API | Auto-emits `LLM` spans → `Correctness` works |
# MAGIC
# MAGIC No glue code needed — the autolog hook attaches span metadata.

# COMMAND ----------

from databricks_langchain import ChatDatabricks, DatabricksVectorSearch

retriever = DatabricksVectorSearch(
    endpoint=VS_ENDPOINT,
    index_name=VS_INDEX_FQN,
    columns=["doc_id", "content"],
).as_retriever(search_kwargs={"k": 3})

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4", temperature=0)

# Quick sanity check
sample = retriever.invoke("What is Z-ordering?")
for d in sample:
    print(d.metadata.get("doc_id"), "—", d.page_content[:80])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build the LangGraph `StateGraph`
# MAGIC
# MAGIC Two nodes, one edge:
# MAGIC
# MAGIC ```
# MAGIC START → retrieve → generate → END
# MAGIC ```
# MAGIC
# MAGIC The graph state carries `question`, `docs`, and `answer`. Each node is a plain Python function — LangGraph simply orchestrates them and gives MLflow a structured trace to record.

# COMMAND ----------

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

class RAGState(TypedDict):
    question: str
    docs: list
    answer: str

def retrieve_node(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    return {**state, "docs": docs}

def generate_node(state: RAGState) -> RAGState:
    context = "\n".join(
        f"[{d.metadata.get('doc_id', 'unknown')}] {d.page_content}"
        for d in state["docs"]
    )
    messages = [
        SystemMessage(content=(
            "Answer using only the provided context. "
            "Cite the doc_id in square brackets after each claim. Be concise."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
    ]
    result = llm.invoke(messages)
    return {**state, "answer": result.content}

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()
print("LangGraph RAG compiled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Wrap the Graph as a `predict_fn`
# MAGIC
# MAGIC `mlflow.genai.evaluate()` expects `predict_fn(question: str) -> str`. The wrapper invokes the graph and returns the final answer string. The `@mlflow.trace` decorator stitches the LangChain spans under one parent trace per row.

# COMMAND ----------

@mlflow.trace
def rag_agent_lg(question: str) -> str:
    out = rag_graph.invoke({"question": question, "docs": [], "answer": ""})
    return out["answer"]

print(rag_agent_lg("What is Z-ordering?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Custom Retrieval Recall Scorer (Reused from Lab 5.2)
# MAGIC
# MAGIC The scorer reads `RETRIEVER` spans from the trace. LangChain autolog tags `DatabricksVectorSearch` invocations with `span_type=RETRIEVER` automatically — same shape the scorer already expects. **Zero changes** to the scorer code.

# COMMAND ----------

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
    return list(dict.fromkeys(hits))

def _doc_ids_from_span(span) -> set[str]:
    """LangChain retriever spans store outputs as a list of Documents (or dicts).
    We accept either shape so the scorer works for raw-Python and LangGraph agents."""
    out = span.outputs
    if not out:
        return set()
    docs = out if isinstance(out, list) else out.get("documents", [])
    ids = set()
    for d in docs or []:
        if isinstance(d, dict):
            meta = d.get("metadata") or {}
            doc_id = meta.get("doc_id") or d.get("doc_id")
        else:
            meta = getattr(d, "metadata", {}) or {}
            doc_id = meta.get("doc_id")
        if doc_id:
            ids.add(doc_id)
    return ids

@scorer
def retrieved_document_recall(inputs, trace) -> Feedback:
    question = inputs.get("question") if isinstance(inputs, dict) else str(inputs)
    expected = set(expected_doc_ids_for(question or ""))
    if not expected:
        return Feedback(
            value=None,
            rationale="No expected_doc_ids known for this question — skipping.",
            source=AssessmentSource(source_type="CODE", source_id="retrieval_recall_v1"),
        )
    retrieved: set[str] = set()
    for span in trace.search_spans(span_type=SpanType.RETRIEVER):
        retrieved |= _doc_ids_from_span(span)
    recall = len(expected & retrieved) / len(expected)
    return Feedback(
        value=round(recall, 2),
        rationale=f"expected={sorted(expected)} retrieved={sorted(retrieved)} recall={recall:.2f}",
        source=AssessmentSource(source_type="CODE", source_id="retrieval_recall_v1"),
    )

print("Recall scorer ready (framework-agnostic).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Run the Three-Pillar Eval

# COMMAND ----------

import mlflow.genai.datasets
from mlflow.genai.scorers import Correctness, RetrievalGroundedness

eval_dataset = mlflow.genai.datasets.get_dataset(name=EVAL_DATASET_FQN)

results_lg = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=rag_agent_lg,
    scorers=[
        Correctness(),
        RetrievalGroundedness(),
        retrieved_document_recall,
    ],
    model_id="models:/rag-agent-langgraph/v1",
)

display(results_lg.tables["eval_results"])

# COMMAND ----------

# DBTITLE 1,Aggregate three-pillar pass rates (LangGraph agent)
display(results_lg.tables["eval_results"].selectExpr(
    "AVG(CASE WHEN `correctness/v1/value`            = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness_rate",
    "AVG(CASE WHEN `retrieval_groundedness/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS groundedness_rate",
    "AVG(`retrieved_document_recall/v1/value`)                                       AS mean_recall",
    "COUNT(*) AS rows",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Framework Parity: LangGraph vs. Raw-OpenAI Agent
# MAGIC
# MAGIC Optional but instructive — if Lab 5.2 has been run in the same experiment, open the experiment UI and **Compare** the runs side-by-side. We expect:
# MAGIC - **Same retrieval recall** (both agents call the same Vector Search index with the same `k`)
# MAGIC - **Comparable correctness** (same model, equivalent prompts; small variance from sampling is normal)
# MAGIC - **Comparable groundedness**
# MAGIC
# MAGIC Material gaps would point at *prompt* differences (system message wording, doc_id formatting), not framework choice.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Diagnose Failure Modes (Same Grid as Lab 5.2)

# COMMAND ----------

# DBTITLE 1,Retrieval misses (groundedness OK, correctness fails, low recall)
display(results_lg.tables["eval_results"].selectExpr(
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

# DBTITLE 1,Hallucinations (high recall, low groundedness)
display(results_lg.tables["eval_results"].selectExpr(
    "request_id", "inputs", "outputs",
    "`retrieval_groundedness/v1/value` AS groundedness",
    "`retrieval_groundedness/v1/rationale` AS groundedness_detail",
    "`retrieved_document_recall/v1/value` AS recall",
).filter(
    "`retrieval_groundedness/v1/value` != 'yes' "
    "AND `retrieved_document_recall/v1/value` >= 0.5"
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | LangGraph `StateGraph` with `retrieve` + `generate` nodes built | ✅ |
# MAGIC | `DatabricksVectorSearch` retriever + `ChatDatabricks` LLM wired in | ✅ |
# MAGIC | `mlflow.langchain.autolog()` produces `RETRIEVER` and `LLM` spans | ✅ |
# MAGIC | Same three judges/scorers as Lab 5.2 ran unchanged | ✅ |
# MAGIC | Failure-mode diagnosis grid reproduced on the LangGraph agent | ✅ |
# MAGIC
# MAGIC **Key takeaway:** The eval contract — `predict_fn` returning a string + a trace with the right span types — is framework-agnostic. Swap the agent implementation, keep the scorers.
# MAGIC
# MAGIC Next: **Lab 5.4b** — same idea for tool-using agents, with LangGraph's `create_react_agent` and the same `ToolCallEfficiency` + `KnowledgeRetention` judges.
