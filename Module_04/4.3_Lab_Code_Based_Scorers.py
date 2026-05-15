# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "1"
# ///
# MAGIC %md
# MAGIC # 🧰 Lab 4.3 — Code-Based Scorers with `@scorer` Decorator
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Pattern 1: Primitive Return** — `@scorer def fn(outputs) -> int` — smallest possible scorer
# MAGIC 2. **Pattern 2: `Feedback` Object** — Return `Feedback(value, rationale, source)` with `AssessmentSource(source_type="CODE")`
# MAGIC 3. **Pattern 3: Trace-Based** — Use `trace.search_spans(span_type=SpanType.CHAT_MODEL)` for latency / span-attribute checks
# MAGIC 4. **Three Lab Scorers** — Build valid-JSON, latency ≤ 5 s, and citation-presence checks
# MAGIC 5. **Mixing With LLM Judges** — Combine code-based scorers and built-in judges in one `evaluate()` call
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 + 2.2 + 4.2
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
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 2 - THE THREE `@SCORER` PATTERNS
# ============================================================================

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
# MAGIC ---
# MAGIC ## Step 3 — Build Lab-Specific Scorers
# MAGIC
# MAGIC We'll write three scorers tied to plausible business rules:
# MAGIC
# MAGIC | Rule | Pattern | Why code, not LLM |
# MAGIC | --- | --- | --- |
# MAGIC | Response is valid JSON | Pattern 2 (`Feedback` with parse error rationale) | Parsing is deterministic — `json.loads` either works or doesn't |
# MAGIC | End-to-end latency ≤ 5 s | Pattern 3 (trace-based) | Latency lives in span timestamps, not output text |
# MAGIC | Response cites at least one doc | Pattern 2 (`Feedback` with regex match) | Regex on `[doc_id]` is faster and cheaper than asking an LLM |
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - BUILD LAB-SPECIFIC SCORERS
# ============================================================================

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
# MAGIC ---
# MAGIC ## Step 4 — Build a Citing Agent
# MAGIC
# MAGIC We instruct the agent to cite `[doc_id]` so the `has_citation` scorer can succeed. Reusing the RAG shape from Lab 4.2.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 4 - BUILD A CITING AGENT
# ============================================================================

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
# MAGIC ---
# MAGIC ## Step 5 — Run All Code-Based Scorers
# MAGIC
# MAGIC `is_valid_json` is expected to fail on this dataset — the agent returns natural language, not JSON. That's fine: it demonstrates a code scorer flagging a contract violation cleanly without an LLM call.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 5 - RUN ALL CODE-BASED SCORERS
# ============================================================================

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[is_valid_json, latency_under_5s, has_citation],
    # model_id="models:/my-agent-citations/v1",
)

display(results.tables["eval_results"])


# COMMAND ----------

# ============================================================================
# ▶️ AGGREGATE CODE-SCORER PASS RATES
# ============================================================================

import pandas as pd

df = results.tables["eval_results"]
agg = pd.DataFrame([{
    "json_valid_rate":   (df["is_valid_json/value"] == "yes").mean(),
    "latency_pass_rate": (df["latency_under_5s/value"] == "yes").mean(),
    "citation_rate":     (df["has_citation/value"] == "yes").mean(),
    "rows":              len(df),
}])
display(agg)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Combine Code Scorers with Built-in Judges
# MAGIC
# MAGIC The whole point: code scorers and LLM judges live in the same `scorers=[...]` list. Use the right tool for each rule.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 6 - COMBINE CODE SCORERS WITH BUILT-IN JUDGES
# ============================================================================

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
    # model_id="models:/my-agent-citations/v1-combined",
)

display(results_combined.tables["eval_results"])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
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
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. When Code Beats LLM
# MAGIC - Any rule that is **exact, deterministic, and cheap** belongs in `@scorer`, not in an LLM judge.
# MAGIC - Examples: JSON validity, schema conformance, length bounds, regex format, latency SLAs, span counts.
# MAGIC - An LLM call to check `json.loads()` works is wasteful — and slower and more flaky.
# MAGIC
# MAGIC ### 2. The Three Patterns
# MAGIC - **Pattern 1 — primitive** for raw numbers/booleans (`int`, `float`, `bool`, `str`).
# MAGIC - **Pattern 2 — `Feedback`** when you want a rationale shown in the UI and explicit `AssessmentSource` provenance.
# MAGIC - **Pattern 3 — trace-based** when the answer lives in spans (latency, retrieval recall, tool-call counts).
# MAGIC
# MAGIC ### 3. Composition Beats Picking One
# MAGIC - Code scorers and LLM judges share one `scorers=[...]` list — use the right tool for each rule.
# MAGIC - Code scorers are usually free (no LLM call) and millisecond-fast — lean on them heavily.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 4.4** — graduate to `make_judge()` for domain-specific graded rubrics.
# MAGIC
