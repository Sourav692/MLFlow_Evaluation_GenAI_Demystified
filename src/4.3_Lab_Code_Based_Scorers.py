# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4.3 — Code-Based Scorers with `@scorer`
# MAGIC
# MAGIC **Goal:** Write deterministic scorers for business rules that LLMs can't reliably check — response format, latency SLAs, citation presence.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Used **Pattern 1 — primitive return** (`@scorer def fn(outputs) -> int`)
# MAGIC 2. Used **Pattern 2 — `Feedback` object** with rationale and `AssessmentSource(source_type="CODE")`
# MAGIC 3. Used **Pattern 3 — trace-based** with `trace.search_spans(span_type=SpanType.CHAT_MODEL)` for latency
# MAGIC 4. Built three lab-specific scorers: **valid JSON**, **latency ≤ 5 s**, **citation presence**
# MAGIC 5. Combined all three with built-in judges in a single `evaluate()` call
# MAGIC
# MAGIC > **When to reach for code-based scorers:** any rule that is exact, deterministic, and cheap. JSON validity is binary — don't pay an LLM judge for it.
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 + 2.2 + 4.2 done.

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
# MAGIC ## Step 2 — The Three `@scorer` Patterns
# MAGIC
# MAGIC ### Pattern 1 — Primitive Return
# MAGIC Smallest possible scorer. Return `int`, `float`, `bool`, or `str` and MLflow stores it directly.
# MAGIC
# MAGIC ### Pattern 2 — `Feedback` Object
# MAGIC When you want a `rationale` shown in the UI and explicit provenance. `AssessmentSource(source_type="CODE")` marks the score as deterministic for downstream filtering.
# MAGIC
# MAGIC ### Pattern 3 — Trace-Based
# MAGIC When the answer to "did this pass?" lives inside the trace, not the output. Latency, retrieval recall@k, tool-call counts, span attributes — all live there.

# COMMAND ----------

from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, AssessmentSource, SpanType

# --- Pattern 1: Primitive return ---
@scorer
def word_count(outputs: str) -> int:
    return len(outputs.split())

# --- Pattern 2: Feedback with rationale ---
@scorer
def has_code_block(outputs: str) -> Feedback:
    passed = "```" in outputs
    return Feedback(
        value="yes" if passed else "no",
        rationale="Response includes a code example" if passed else "No code block found",
        source=AssessmentSource(source_type="CODE", source_id="code_checker_v1"),
    )

# --- Pattern 3: Trace-based latency check ---
@scorer
def latency_ok(trace) -> Feedback:
    spans = trace.search_spans(span_type=SpanType.CHAT_MODEL)
    if not spans:
        return Feedback(value="no", rationale="No CHAT_MODEL span found in trace")
    span = spans[0]
    ms = (span.end_time_ns - span.start_time_ns) / 1e6
    return Feedback(
        value="yes" if ms <= 5000 else "no",
        rationale=f"{ms:.0f}ms (SLA: 5000ms)",
        source=AssessmentSource(source_type="CODE", source_id="latency_v1"),
    )

print("Three reference scorers defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build Lab-Specific Scorers
# MAGIC
# MAGIC We'll write three scorers tied to plausible business rules:
# MAGIC
# MAGIC | Rule | Pattern | Why code, not LLM |
# MAGIC | --- | --- | --- |
# MAGIC | Response is valid JSON | Pattern 2 (`Feedback` with parse error rationale) | Parsing is deterministic — `json.loads` either works or doesn't |
# MAGIC | End-to-end latency ≤ 5 s | Pattern 3 (trace-based) | Latency lives in span timestamps, not output text |
# MAGIC | Response cites at least one doc | Pattern 2 (`Feedback` with regex match) | Regex on `[doc_id]` is faster and cheaper than asking an LLM |

# COMMAND ----------

import json
import re
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, AssessmentSource, SpanType

@scorer
def is_valid_json(outputs: str) -> Feedback:
    try:
        json.loads(outputs)
        return Feedback(
            value="yes",
            rationale="Response parses as JSON",
            source=AssessmentSource(source_type="CODE", source_id="json_validity_v1"),
        )
    except (json.JSONDecodeError, TypeError) as e:
        return Feedback(
            value="no",
            rationale=f"Parse error: {e}",
            source=AssessmentSource(source_type="CODE", source_id="json_validity_v1"),
        )

@scorer
def latency_under_5s(trace) -> Feedback:
    spans = trace.search_spans(span_type=SpanType.CHAT_MODEL)
    if not spans:
        return Feedback(value="no", rationale="No CHAT_MODEL span")
    ms = (spans[0].end_time_ns - spans[0].start_time_ns) / 1e6
    return Feedback(
        value="yes" if ms <= 5000 else "no",
        rationale=f"{ms:.0f}ms",
        source=AssessmentSource(source_type="CODE", source_id="latency_5s_v1"),
    )

CITATION_RE = re.compile(r"\[[^\]]+\]")

@scorer
def has_citation(outputs: str) -> Feedback:
    matches = CITATION_RE.findall(outputs)
    return Feedback(
        value="yes" if matches else "no",
        rationale=(f"Found citations: {matches}" if matches else "No [doc_id] citations in response"),
        source=AssessmentSource(source_type="CODE", source_id="citation_v1"),
    )

print("Lab-specific scorers defined: is_valid_json, latency_under_5s, has_citation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Build a Citing Agent
# MAGIC
# MAGIC We instruct the agent to cite `[doc_id]` so the `has_citation` scorer can succeed. Reusing the RAG shape from Lab 4.2.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

DOCS = {
    "delta lake":     "Delta Lake provides ACID transactions, schema enforcement, and time travel.",
    "z-order":        "Z-ordering co-locates related data to improve data skipping.",
    "vacuum":         "VACUUM removes data files older than the retention threshold.",
    "unity catalog":  "Unity Catalog provides centralized governance with column-level lineage.",
}

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str) -> list[dict]:
    q = question.lower()
    return [{"page_content": v, "metadata": {"doc_id": k}} for k, v in DOCS.items() if k in q] \
           or [{"page_content": "No matching docs.", "metadata": {"doc_id": "none"}}]

@mlflow.trace
def my_agent(question: str) -> str:
    docs = retrieve(question)
    context = "\n".join(f"[{d['metadata']['doc_id']}] {d['page_content']}" for d in docs)
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": (
                "Answer using only the provided context. Cite each fact with [doc_id]. Be concise."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content

print(my_agent("What is Z-ordering?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Run All Code-Based Scorers
# MAGIC
# MAGIC `is_valid_json` is expected to fail on this dataset — the agent returns natural language, not JSON. That's fine: it demonstrates a code scorer flagging a contract violation cleanly without an LLM call.

# COMMAND ----------

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[is_valid_json, latency_under_5s, has_citation],
    model_id="models:/my-agent-citations/v1",
)

display(results.tables["eval_results"])

# COMMAND ----------

# DBTITLE 1,Aggregate code-scorer pass rates
display(results.tables["eval_results"].selectExpr(
    "AVG(CASE WHEN `is_valid_json/v1/value`     = 'yes' THEN 1.0 ELSE 0.0 END) AS json_valid_rate",
    "AVG(CASE WHEN `latency_under_5s/v1/value`  = 'yes' THEN 1.0 ELSE 0.0 END) AS latency_pass_rate",
    "AVG(CASE WHEN `has_citation/v1/value`      = 'yes' THEN 1.0 ELSE 0.0 END) AS citation_rate",
    "COUNT(*) AS rows",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Combine Code Scorers with Built-in Judges
# MAGIC
# MAGIC The whole point: code scorers and LLM judges live in the same `scorers=[...]` list. Use the right tool for each rule.

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RelevanceToQuery

results_combined = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[
        # LLM judges for semantic quality
        Correctness(),
        RelevanceToQuery(),
        # Code scorers for deterministic rules
        latency_under_5s,
        has_citation,
    ],
    model_id="models:/my-agent-citations/v1-combined",
)

display(results_combined.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | Pattern 1 (primitive return) demonstrated via `word_count` | ✅ |
# MAGIC | Pattern 2 (`Feedback` + `AssessmentSource`) used in 3 scorers | ✅ |
# MAGIC | Pattern 3 (trace-based) used for latency check | ✅ |
# MAGIC | JSON validity, latency ≤ 5 s, citation presence all running | ✅ |
# MAGIC | Code scorers + LLM judges combined in one `evaluate()` call | ✅ |
# MAGIC
# MAGIC Next: **Lab 4.4** — build a custom LLM judge with `make_judge()` for domain-specific rubrics.
