# Databricks notebook source
# MAGIC %md
# MAGIC # 🧮 Capstone 8.3 — The Full Scorer Suite
# MAGIC
# MAGIC **Goal:** Assemble the **complete, capstone-grade scorer suite** that touches every quality pillar at once: built-in (`Correctness`, `RetrievalGroundedness`, `Safety`), `Guidelines` for compliance rules, a custom code `@scorer` for latency, and a `make_judge` LLM judge for UC technical accuracy. Smoke-test the suite end-to-end against the dataset from 8.2 and freeze the resulting artefact dictionary that Notebooks 4, 5, and 6 import.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will:
# MAGIC 1. **Build the Suite** — Compose `Correctness` + `RetrievalGroundedness` + `Safety` + `Guidelines` + `@scorer` latency + `make_judge` UC accuracy in one list
# MAGIC 2. **Custom Code Scorer** — Implement a `@scorer` that reads `trace.info.execution_time_ms` and fails rows over a configurable budget
# MAGIC 3. **Custom LLM Judge** — Author a `make_judge` rubric covering UC-specific technical accuracy with explicit 1-5 scoring
# MAGIC 4. **Compliance Rules** — Encode three `Guidelines` (English, no destructive SQL, length cap) as pass/fail policy
# MAGIC 5. **Smoke-Test End-to-End** — Run the full suite against `tutorial_capstone_eval_v1` and confirm every column appears in `results.tables`
# MAGIC 6. **Freeze the Artefact** — Log scorer source code + `SCORER_NAMES` mapping so downstream gates reference exact column names
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Capstones 8.1 and 8.2 complete (RAG endpoint live + UC eval dataset registered)
# MAGIC - Familiarity with Module 4 (built-in judges, `@scorer`, `make_judge`, `Guidelines`)
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
# MAGIC ## Step 1 — Capstone Constants & Experiment
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧭 STEP 1 - CAPSTONE CONSTANTS & EXPERIMENT
# ============================================================================

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_08_capstone"
EVAL_DATASET_FQN  = f"{CATALOG}.{SCHEMA}.tutorial_capstone_eval_v1"
SERVING_ENDPOINT  = "genai-capstone-rag"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.openai.autolog()

print(f"🎯 Eval dataset    : {EVAL_DATASET_FQN}")
print(f"🚀 Endpoint        : {SERVING_ENDPOINT}")
print(f"🧪 Experiment      : {EXPERIMENT_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Built-In Judges: Correctness, RetrievalGroundedness, Safety
# MAGIC
# MAGIC These three are the spine of three-pillar evaluation: `Correctness` reads the final answer, `RetrievalGroundedness` reads the RETRIEVER span outputs, `Safety` checks the response for harmful content.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🛠️ STEP 2 - BUILT-IN JUDGES
# ============================================================================

from mlflow.genai.scorers import (
    Correctness,
    RetrievalGroundedness,
    Safety,
    Guidelines,
    scorer,
)

correctness  = Correctness()
groundedness = RetrievalGroundedness()
safety       = Safety()

print("✅ Built-in judges initialised: Correctness, RetrievalGroundedness, Safety")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — `Guidelines` for Compliance Rules
# MAGIC
# MAGIC `Guidelines` runs a single LLM call that scores the response against a list of pass/fail rules. Rules become **policy artefacts** — they belong in code review, not in someone's head.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📜 STEP 3 - COMPLIANCE GUIDELINES
# ============================================================================

compliance = Guidelines(
    name="compliance",
    guidelines=[
        "Response must be in English.",
        "Response must not include or suggest DROP TABLE, DELETE FROM, or other destructive SQL.",
        "Response must be no longer than 8 sentences.",
        "Response must cite at least one Unity Catalog concept (catalog, schema, grant, lineage, mask, audit, share, volume, model, view) when the question is about UC.",
    ],
)

print("✅ Compliance Guidelines judge ready.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Custom Code `@scorer`: Latency Budget
# MAGIC
# MAGIC A pure-Python scorer reads `trace.info.execution_time_ms` and fails the row if it exceeds a configurable budget. Code scorers are deterministic, free, and instant — perfect for SLO-style checks that don't need an LLM.
# MAGIC

# COMMAND ----------

# ============================================================================
# ⏱️ STEP 4 - LATENCY @scorer
# ============================================================================

from mlflow.entities import Feedback, AssessmentSource

LATENCY_BUDGET_MS = 8_000   # 8 s end-to-end SLO for UC governance Q&A

@scorer
def latency_under_budget(trace) -> Feedback:
    """Pass when total trace execution time is under LATENCY_BUDGET_MS."""
    elapsed = getattr(trace.info, "execution_time_ms", None)
    if elapsed is None:
        return Feedback(
            value=None,
            rationale="trace.info.execution_time_ms missing — cannot evaluate latency.",
            source=AssessmentSource(source_type="CODE", source_id="latency_v1"),
        )
    passed = elapsed <= LATENCY_BUDGET_MS
    return Feedback(
        value="yes" if passed else "no",
        rationale=f"Elapsed {elapsed} ms vs budget {LATENCY_BUDGET_MS} ms.",
        source=AssessmentSource(source_type="CODE", source_id="latency_v1"),
    )

print(f"✅ latency_under_budget @scorer ready (budget = {LATENCY_BUDGET_MS} ms)")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Custom LLM Judge: UC Technical Accuracy via `make_judge`
# MAGIC
# MAGIC Built-in `Correctness` checks alignment to `expected_facts` but doesn't test depth. We add a graded UC-specific rubric — 1 to 5 — that rewards precision and important caveats (retention windows, propagation rules, ANSI grants).
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧑‍⚖️ STEP 5 - UC ACCURACY make_judge
# ============================================================================

from mlflow.genai.judges import make_judge

uc_accuracy = make_judge(
    name="uc_technical_accuracy",
    instructions="""
Evaluate whether {{ outputs }} is technically accurate as an answer to the
Unity Catalog question in {{ inputs }}.

Score 1-5:
  5 = Precise, correct ANSI grant model, includes important caveats (retention
      defaults, propagation rules, system-table immutability, alias vs stage)
  4 = Mostly correct, minor omissions
  3 = Partially correct or missing key caveats
  2 = Significant inaccuracy or misleading
  1 = Fundamentally wrong / hallucinated APIs (e.g. invented commands)

Penalise: invented commands, wrong namespace levels, suggesting destructive
SQL, or claiming features that don't exist in Unity Catalog.

Reason briefly. Return the integer score on the final line.
""",
    model="databricks:/databricks-claude-opus-4-6",
)

print("✅ uc_technical_accuracy judge ready.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Compose the Full Suite
# MAGIC
# MAGIC The order in `SCORERS` doesn't change semantics, but keeping a stable ordering helps when reading `results.metrics` columns.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧪 STEP 6 - COMPOSE FULL SCORER SUITE
# ============================================================================

SCORERS = [
    correctness,
    groundedness,
    safety,
    compliance,
    latency_under_budget,
    uc_accuracy,
]

SCORER_NAMES = {
    "correctness":           "correctness",
    "retrieval_groundedness":"retrieval_groundedness",
    "safety":                "safety",
    "compliance":            "compliance",
    "latency":               "latency_under_budget",
    "uc_accuracy":           "uc_technical_accuracy",
}

for name in SCORER_NAMES.values():
    print(f"  • {name}")
print(f"\n✅ {len(SCORERS)} scorers in the capstone suite.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — `predict_fn` Pointing at the Live Endpoint
# MAGIC
# MAGIC `mlflow.genai.scorers.to_predict_fn("endpoints:/...")` returns a function that calls the deployed RAG endpoint. Eval rows flow in, traces flow out — and Steps 4-6 read those traces.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🚀 STEP 7 - predict_fn FROM ENDPOINT
# ============================================================================

from mlflow.genai.scorers import to_predict_fn

predict_fn = to_predict_fn(f"endpoints:/{SERVING_ENDPOINT}")
print(f"✅ predict_fn bound to endpoints:/{SERVING_ENDPOINT}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Smoke-Test the Full Suite
# MAGIC
# MAGIC Run a single `mlflow.genai.evaluate(...)` call wiring together the dataset (8.2), endpoint (8.1), and full scorer suite (this notebook). Inspect the per-row table to confirm every scorer column shows up.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔬 STEP 8 - SMOKE-TEST FULL SUITE
# ============================================================================

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=EVAL_DATASET_FQN)

with mlflow.start_run(run_name="capstone-scorer-suite-smoke") as run:
    mlflow.set_tags({"capstone": "module_08", "layer": "scorers", "phase": "smoke"})
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=SCORERS,
        model_id=f"endpoints:/{SERVING_ENDPOINT}",
    )
    SUITE_RUN_ID = run.info.run_id

print(f"✅ Smoke-test run: {SUITE_RUN_ID}")
print("\nAggregate metrics:")
for k, v in sorted(results.metrics.items()):
    print(f"  {k}: {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9 — Inspect Per-Row Results
# MAGIC
# MAGIC Eyeball the per-row table to confirm each scorer column is populated and rationales make sense. Anomaly hunting at this stage is faster than discovering it during the gate run in Notebook 4.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 9 - PER-ROW RESULTS
# ============================================================================

display(results.tables["eval_results"])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 10 — Freeze Scorer Names + Source for Downstream Gates
# MAGIC
# MAGIC Notebook 4's `THRESHOLDS` dict references **exact column names** (`<scorer_name>/mean`, `<scorer_name>/ratio_pass`). Logging the mapping here is the contract.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📝 STEP 10 - FREEZE SCORER ARTEFACTS
# ============================================================================

with mlflow.start_run(run_id=SUITE_RUN_ID):
    mlflow.log_dict(SCORER_NAMES, "scorer_names.json")
    mlflow.log_param("scorer_count",       len(SCORERS))
    mlflow.log_param("latency_budget_ms",  LATENCY_BUDGET_MS)

try:
    dbutils.jobs.taskValues.set(key="latency_budget_ms", value=LATENCY_BUDGET_MS)
    print("✅ taskValue 'latency_budget_ms' set.")
except Exception:
    print("ℹ️  Interactive run — taskValue skipped.")

print("\nFinal scorer name → column mapping:")
for k, v in SCORER_NAMES.items():
    print(f"  {k:22s} → {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | `Correctness` + `RetrievalGroundedness` + `Safety` initialised | ✅ |
# MAGIC | `Guidelines` compliance judge with four rules | ✅ |
# MAGIC | Custom `@scorer` enforcing the latency SLO | ✅ |
# MAGIC | `make_judge` UC technical accuracy rubric (1-5) | ✅ |
# MAGIC | Full suite smoke-tested against the live endpoint | ✅ |
# MAGIC | Per-row table shows every scorer column populated | ✅ |
# MAGIC | `SCORER_NAMES` mapping logged for the gate | ✅ |
# MAGIC
# MAGIC **Next:** *Capstone 8.4* — wire the suite into a **CI quality gate** that fails the Workflow when any threshold drops.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we built the **Scorer layer** of the capstone:
# MAGIC
# MAGIC ### 1. The Suite Covers Every Pillar at Once
# MAGIC - **Generation**: `Correctness` + `uc_technical_accuracy` (depth + caveats).
# MAGIC - **Retrieval**: `RetrievalGroundedness` reads RETRIEVER span outputs.
# MAGIC - **Policy**: `Guidelines` for compliance + `Safety` for harm.
# MAGIC - **SLO**: `latency_under_budget` from trace timings.
# MAGIC
# MAGIC ### 2. Code + LLM Scorers Are Complementary
# MAGIC - Code scorers (latency) are deterministic and free — use them for hard SLO checks.
# MAGIC - LLM judges (`make_judge`, `Guidelines`) catch nuance — use them for accuracy and policy.
# MAGIC
# MAGIC ### 3. Names Are the Contract
# MAGIC - Notebook 4's gate references `<scorer_name>/mean` and `<scorer_name>/ratio_pass` literally — `SCORER_NAMES` makes the contract auditable.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Move to **Capstone 8.4** to put this suite behind a `dbutils.notebook.exit("FAILED")` gate wired into a Databricks Workflow.
# MAGIC