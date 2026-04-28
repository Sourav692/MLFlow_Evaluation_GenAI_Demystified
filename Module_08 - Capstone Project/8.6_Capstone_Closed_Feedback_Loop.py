# Databricks notebook source
# MAGIC %md
# MAGIC # 🔁 Capstone 8.6 — Closed Human Feedback Loop
# MAGIC
# MAGIC **Goal:** Close the capstone loop. Simulate users sending **human feedback** on production traces via `mlflow.log_feedback(...)`, compare those labels against the registered LLM judges (8.5), iterate the judge prompt where they disagree, and **re-run the gate (8.4) with the improved judges** to confirm the closed-loop is auditable end-to-end.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will:
# MAGIC 1. **Pick a Trace Cohort** — Pull recent traces from 8.5 with their automated assessments
# MAGIC 2. **Simulate Human Feedback** — `mlflow.log_feedback(trace_id=..., name="human_correctness", ...)` for ~20 traces
# MAGIC 3. **Agreement Analysis** — Build a confusion matrix between the LLM judge `correctness` and the human label
# MAGIC 4. **Diagnose Disagreement** — Inspect rows where the judge said *yes* but humans said *no* (and vice versa)
# MAGIC 5. **Iterate the Judge Prompt** — Tighten `make_judge` instructions to address the failure modes you found
# MAGIC 6. **Re-Run the Gate** — Plug the improved judge back into the suite and re-run `mlflow.genai.evaluate(...)`
# MAGIC 7. **Capstone Outcome** — Confirm the full enterprise eval system is closed-loop and governed end-to-end
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Capstones 8.1-8.5 complete (endpoint + gate + registered scorers + simulated traffic)
# MAGIC - Permission to write `Assessment` rows to traces in this experiment
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
# 🧭 STEP 1 - CAPSTONE CONSTANTS
# ============================================================================

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_08_capstone"
EVAL_DATASET_FQN = f"{CATALOG}.{SCHEMA}.tutorial_capstone_eval_v1"
SERVING_ENDPOINT = "genai-capstone-rag"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"🎯 Eval dataset : {EVAL_DATASET_FQN}")
print(f"🚀 Endpoint     : {SERVING_ENDPOINT}")
print(f"🧪 Experiment   : {EXPERIMENT_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Pull a Trace Cohort to Label
# MAGIC
# MAGIC We pick the most recent ~20 traces with their automated assessments from 8.5. In a real workflow, these come from a labelling app (Streamlit, Databricks Apps, etc.) where SMEs mark answers as good / bad / borderline.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🪧 STEP 2 - PICK TRACES TO LABEL
# ============================================================================

traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    max_results=20,
    order_by=["timestamp_ms DESC"],
)

print(f"Pulled {len(traces)} traces.")
if len(traces) == 0:
    print("⚠️  No traces — re-run 8.5 Step 4 first to generate production traffic.")
else:
    display(traces.head(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Simulate Human Feedback via `mlflow.log_feedback`
# MAGIC
# MAGIC Real users press 👍 / 👎 in your app; we'll simulate that here with a deterministic policy: short or off-topic answers get a 👎, the rest get a 👍. The point is the **API shape** — `log_feedback` writes an `Assessment` of `source_type="HUMAN"` onto the trace, side-by-side with the LLM judges.
# MAGIC

# COMMAND ----------

# ============================================================================
# ✍️ STEP 3 - SIMULATE HUMAN FEEDBACK
# ============================================================================

from mlflow.entities import AssessmentSource

def synthetic_human_label(answer: str, question: str) -> tuple[str, str]:
    """Returns (value, rationale) — would be a real SME judgement in production."""
    if not answer or len(answer) < 80:
        return "no", "Too short / not enough detail."
    if any(t in (question or "").lower() for t in ["weather", "recipe", "chickpea"]):
        return "no", "Off-topic question, model should have refused."
    if "unity catalog" in (answer or "").lower() or "delta" in (answer or "").lower():
        return "yes", "Cites a UC concept and addresses the question."
    return "no", "Missing UC-specific terminology."

labelled = 0
for tr in traces.itertuples():
    trace_id = getattr(tr, "trace_id", None) or getattr(tr, "request_id", None)
    if trace_id is None:
        continue
    inputs   = getattr(tr, "request",  None) or {}
    outputs  = getattr(tr, "response", None) or ""
    question = inputs.get("question") if isinstance(inputs, dict) else str(inputs)
    answer   = outputs if isinstance(outputs, str) else str(outputs)

    value, rationale = synthetic_human_label(answer, question)
    try:
        mlflow.log_feedback(
            trace_id=trace_id,
            name="human_correctness",
            value=value,
            rationale=rationale,
            source=AssessmentSource(source_type="HUMAN", source_id=USER_EMAIL),
        )
        labelled += 1
    except Exception as e:
        print(f"⚠️  log_feedback failed for {trace_id}: {e!r}")

print(f"✅ Logged human_correctness feedback on {labelled} traces.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Build Judge-vs-Human Agreement
# MAGIC
# MAGIC Pull traces again, line up the registered `capstone_correctness` (LLM judge) value with the new `human_correctness` value, and compute the agreement rate + confusion matrix. This is the calibration score — anything below ~0.85 means the judge is not yet trustworthy enough to gate on by itself.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📐 STEP 4 - JUDGE vs HUMAN AGREEMENT
# ============================================================================

import time, pandas as pd

time.sleep(15)  # let log_feedback land

labelled_traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    max_results=50,
    order_by=["timestamp_ms DESC"],
)

rows = []
for tr in labelled_traces.itertuples():
    by_name = {}
    for a in (getattr(tr, "assessments", []) or []):
        if hasattr(a, "name"):
            by_name[a.name] = getattr(getattr(a, "feedback", None), "value", None)
    if "human_correctness" in by_name:
        rows.append({
            "trace_id":          getattr(tr, "trace_id", None),
            "judge_correctness": by_name.get("capstone_correctness") or by_name.get("correctness"),
            "human_correctness": by_name.get("human_correctness"),
        })

if not rows:
    print("⚠️  No rows with both judge + human assessments yet — wait and re-run.")
else:
    df = pd.DataFrame(rows).dropna()
    df["agree"] = df["judge_correctness"] == df["human_correctness"]
    print(f"Pairs with both labels : {len(df)}")
    print(f"Agreement rate          : {df['agree'].mean():.2%}")
    confusion = (
        df.groupby(["judge_correctness", "human_correctness"])
          .size()
          .reset_index(name="count")
    )
    display(confusion)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Diagnose Disagreement
# MAGIC
# MAGIC The interesting cells are **judge=yes / human=no** (false confidence) and **judge=no / human=yes** (false alarm). Read the rationales, find a pattern, and turn it into a tightening rule for the next judge prompt.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔬 STEP 5 - DIAGNOSE DISAGREEMENT
# ============================================================================

if 'df' in globals() and len(df):
    disagreements = df[~df["agree"]]
    print(f"Disagreement rows: {len(disagreements)}")
    display(disagreements)
else:
    print("ℹ️  No agreement DataFrame yet — run Step 4 first.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Iterate the Judge Prompt
# MAGIC
# MAGIC Encode what you learned in Step 5 into a tightened `make_judge`. Common iteration: penalise off-topic answers explicitly, require a UC concept to be cited, refuse to score 5 without a caveat. This judge is the one we'll re-run the gate against.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🛠️ STEP 6 - V2 JUDGE PROMPT
# ============================================================================

from mlflow.genai.judges import make_judge

uc_accuracy_v2 = make_judge(
    name="uc_technical_accuracy_v2",
    instructions="""
Evaluate whether {{ outputs }} is technically accurate as an answer to the
Unity Catalog question in {{ inputs }}.

Hard rules (failures override the rubric):
- If the question is off-topic (weather, recipes, etc.) and the answer engages
  rather than refusing, score 1.
- If the answer is shorter than ~80 characters AND the question is non-trivial,
  cap the score at 2.
- If the answer does not cite at least one Unity Catalog concept (catalog,
  schema, grant, lineage, mask, audit, share, volume, model, view, alias),
  cap the score at 3.

Otherwise, grade 1-5:
  5 = Precise + correct caveats (retention, propagation, system-table immutability)
  4 = Mostly correct, minor omissions
  3 = Partially correct or missing key caveats
  2 = Significant inaccuracy
  1 = Fundamentally wrong / hallucinated APIs

Reason briefly. Return the integer score on the final line.
""",
    model="databricks:/databricks-claude-opus-4-6",
)

print("✅ uc_technical_accuracy_v2 ready (off-topic + length + UC-citation rules).")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Re-Run the Gate With the Improved Judge
# MAGIC
# MAGIC Drop `uc_technical_accuracy_v2` into the suite alongside the existing scorers and re-run the eval. We log the run with `phase=feedback_loop` so the audit trail shows v1 → v2 progression.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🚦 STEP 7 - RE-RUN GATE WITH v2 JUDGE
# ============================================================================

from mlflow.entities import Feedback
from mlflow.genai.scorers import (
    Correctness, RetrievalGroundedness, Safety, Guidelines, scorer, to_predict_fn,
)
import mlflow.genai.datasets

LATENCY_BUDGET_MS = 8_000

@scorer
def latency_under_budget(trace) -> Feedback:
    elapsed = getattr(trace.info, "execution_time_ms", None)
    if elapsed is None:
        return Feedback(value=None, rationale="latency missing",
                        source=AssessmentSource(source_type="CODE", source_id="latency_v1"))
    passed = elapsed <= LATENCY_BUDGET_MS
    return Feedback(
        value="yes" if passed else "no",
        rationale=f"Elapsed {elapsed} ms vs budget {LATENCY_BUDGET_MS} ms.",
        source=AssessmentSource(source_type="CODE", source_id="latency_v1"),
    )

compliance = Guidelines(
    name="compliance",
    guidelines=[
        "Response must be in English.",
        "Response must not include or suggest DROP TABLE, DELETE FROM, or other destructive SQL.",
        "Response must be no longer than 8 sentences.",
    ],
)

SCORERS_V2 = [
    Correctness(),
    RetrievalGroundedness(),
    Safety(),
    compliance,
    latency_under_budget,
    uc_accuracy_v2,
]

predict_fn   = to_predict_fn(f"endpoints:/{SERVING_ENDPOINT}")
eval_dataset = mlflow.genai.datasets.get_dataset(name=EVAL_DATASET_FQN)

with mlflow.start_run(run_name="capstone-feedback-loop-v2-eval") as run:
    mlflow.set_tags({
        "capstone":     "module_08",
        "layer":        "feedback_loop",
        "judge_version": "v2",
        "phase":        "feedback_loop",
    })
    results_v2 = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=SCORERS_V2,
        model_id=f"endpoints:/{SERVING_ENDPOINT}",
    )
    LOOP_RUN_ID = run.info.run_id

print("✅ v2 eval complete")
print("\nv2 metrics:")
for k, v in sorted(results_v2.metrics.items()):
    print(f"  {k}: {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Confirm the v2 Gate Verdict
# MAGIC
# MAGIC Re-apply the same `THRESHOLDS` policy (with the v2 judge column name) and confirm the gate verdict. If the judge tightening uncovered new failures, the gate now blocks them — exactly the behaviour we want from a closed loop.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🚦 STEP 8 - v2 GATE VERDICT
# ============================================================================

THRESHOLDS_V2 = {
    "correctness/mean":                   3.5,
    "retrieval_groundedness/ratio_pass":  0.90,
    "safety/ratio_pass":                  1.0,
    "compliance/ratio_pass":              0.90,
    "latency_under_budget/ratio_pass":    0.95,
    "uc_technical_accuracy_v2/mean":      3.0,
}

def evaluate_gate(metrics, thresholds):
    failures = []
    for k, floor in thresholds.items():
        v = metrics.get(k)
        if v is None:
            failures.append(f"{k}: MISSING")
        elif v < floor:
            failures.append(f"{k}: {v:.3f} < {floor}")
    return failures

v2_failures = evaluate_gate(results_v2.metrics, THRESHOLDS_V2)

with mlflow.start_run(run_id=LOOP_RUN_ID):
    mlflow.log_dict(THRESHOLDS_V2,                                              "thresholds.json")
    mlflow.log_dict({"failures": v2_failures, "passed": not v2_failures},        "gate_verdict.json")
    mlflow.set_tag("gate_passed", str(not v2_failures).lower())

if v2_failures:
    print("❌ v2 GATE FAILED — improved judge surfaced regressions:")
    for f in v2_failures:
        print(" •", f)
else:
    print("✅ v2 gate passed — closed loop confirmed.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9 — Capstone Loop Audit
# MAGIC
# MAGIC A single MLflow query gives the auditor the full closed loop in one table: app deploy → dataset creation → scorer suite → gate runs (v1 + v2) → monitoring → feedback loop.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 9 - CAPSTONE LOOP AUDIT
# ============================================================================

audit = mlflow.search_runs(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    filter_string="tags.capstone = 'module_08'",
    order_by=["start_time ASC"],
    max_results=50,
)
display(audit)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | Trace cohort labelled with `mlflow.log_feedback(name="human_correctness")` | ✅ |
# MAGIC | Judge-vs-human agreement rate + confusion matrix computed | ✅ |
# MAGIC | Disagreement rows inspected to extract failure-mode rules | ✅ |
# MAGIC | `uc_technical_accuracy_v2` judge re-authored with off-topic / length / UC-citation rules | ✅ |
# MAGIC | Gate re-run with v2 suite; verdict logged with `gate_passed` tag | ✅ |
# MAGIC | Single MLflow audit query stitches together every layer of Module 8 | ✅ |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🏆 Module 8 Capstone — Final Outcome
# MAGIC
# MAGIC You have built and can operate a **complete enterprise-grade MLflow GenAI evaluation system on Databricks** — covering:
# MAGIC
# MAGIC | Layer | Notebook | What you built |
# MAGIC | --- | --- | --- |
# MAGIC | **App** | 8.1 | RAG agent on Vector Search, deployed to Model Serving with AI Gateway + inference table |
# MAGIC | **Dataset** | 8.2 | UC-backed `tutorial_capstone_eval_v1` from synthetic + production + edge cases |
# MAGIC | **Scorers** | 8.3 | Built-in + Guidelines + custom `@scorer` + `make_judge` in one suite |
# MAGIC | **Gate** | 8.4 | `dbutils.notebook.exit("QUALITY_GATE_FAILED")` Workflow gate with versioned thresholds |
# MAGIC | **Monitoring** | 8.5 | Registered scorers running async at sample rates against live traffic |
# MAGIC | **Feedback Loop** | 8.6 | Human feedback → judge agreement → judge iteration → re-gate |
# MAGIC
# MAGIC **Every layer is governed by Unity Catalog and auditable end-to-end:**
# MAGIC - The model is registered to UC.
# MAGIC - The eval dataset is registered to UC.
# MAGIC - The inference table lives in UC.
# MAGIC - Each gate run logs `thresholds.json` + `gate_verdict.json` + a `gate_passed` tag.
# MAGIC - Each judge change logs a new MLflow run with `tags.capstone = 'module_08'`.
# MAGIC
# MAGIC End-to-end:
# MAGIC ```
# MAGIC deploy → gateway → traces → registered judges → human feedback → judge iteration → gate → next deploy
# MAGIC ```
# MAGIC
# MAGIC That's the closed loop. **The MLflow GenAI evaluation system is now production-grade.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we built the **Feedback Loop** of the capstone:
# MAGIC
# MAGIC ### 1. Human Feedback Is an Assessment Like Any Other
# MAGIC - `mlflow.log_feedback(...)` writes an `Assessment` of `source_type="HUMAN"` onto the trace next to the LLM judge.
# MAGIC - Same query surface (`search_traces`) means agreement analysis is just a join.
# MAGIC
# MAGIC ### 2. Disagreement Is the Signal to Iterate
# MAGIC - The cells where judge ≠ human are the failure-mode catalogue for the next judge prompt.
# MAGIC - Encode hard rules ("refuse off-topic", "require a UC concept") as overrides at the top of the rubric.
# MAGIC
# MAGIC ### 3. The Loop Is the Product
# MAGIC - v2 judge → re-run gate → new `gate_verdict.json` → new tag.
# MAGIC - The audit query in Step 9 is the artefact compliance teams want — every change in one searchable table.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Promote the v2 judge to the registered scorer in 8.5 (lower the v1 sample rate, start v2 at 0.2).
# MAGIC - Wire 8.4 into a Databricks Workflow that runs nightly against a freshly merged production sample from 8.2.
# MAGIC - Repeat the loop: humans label → judges tighten → gate gets sharper.
# MAGIC