# Databricks notebook source
# MAGIC %md
# MAGIC # 📡 Capstone 8.5 — Production Monitoring with Registered Scorers
# MAGIC
# MAGIC **Goal:** Once the gate has approved a deploy (8.4), continuously score live traffic. We **register** the capstone's LLM judges + custom scorers against the deployed endpoint at configured sample rates, simulate ~50 production requests, and verify that judge feedback appears on traces and the inference table is recording everything.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will:
# MAGIC 1. **Register Scorers** — Attach `Correctness`, `RetrievalGroundedness`, `Safety`, `Guidelines`, and the `make_judge` UC accuracy judge to the live endpoint via `.register().start()` with per-scorer sample rates
# MAGIC 2. **Tune Sample Rates** — Higher rates for cheap deterministic judges, lower for expensive LLM judges
# MAGIC 3. **Simulate Production Traffic** — Send 50 governance questions through the endpoint to populate traces
# MAGIC 4. **Verify Feedback on Traces** — Use `mlflow.search_traces` to confirm judges attached `Assessment` rows to recent traces
# MAGIC 5. **Inference Table Health** — Query the AI Gateway inference table for status code mix, latency P95, and gateway block count
# MAGIC 6. **Aggregate Production Quality** — Compute a daily roll-up: % blocked, mean correctness, latency P95 — the dashboard you'd page on
# MAGIC 7. **Handle Drift** — Sketch how to lower a sample rate, swap a judge, or pause monitoring without redeploying
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Capstones 8.1-8.4 complete (endpoint live, gate green, dataset registered)
# MAGIC - `mlflow.genai.scorers.register` available (MLflow 3.1+)
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
# MAGIC ## Step 1 — Capstone Constants
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧭 STEP 1 - CAPSTONE CONSTANTS
# ============================================================================

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_08_capstone"

SERVING_ENDPOINT = "genai-capstone-rag"
INFERENCE_TABLE  = f"{CATALOG}.{SCHEMA}.capstone_rag_payload_request_logs"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"🚀 Endpoint        : {SERVING_ENDPOINT}")
print(f"🗃️  Inference table : {INFERENCE_TABLE}")
print(f"🧪 Experiment      : {EXPERIMENT_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Define the Production-Side Scorers
# MAGIC
# MAGIC Same suite as the gate, but tuned for **continuous** runs. We re-instantiate them here (a registered scorer is bound to its definition at registration time) and keep judges that are too expensive for every request out of the registration list — those stay offline-only.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧮 STEP 2 - PRODUCTION SCORERS
# ============================================================================

from mlflow.entities import Feedback, AssessmentSource
from mlflow.genai.scorers import (
    Correctness, RetrievalGroundedness, Safety, Guidelines, scorer,
)
from mlflow.genai.judges import make_judge

compliance = Guidelines(
    name="compliance",
    guidelines=[
        "Response must be in English.",
        "Response must not include or suggest DROP TABLE, DELETE FROM, or other destructive SQL.",
        "Response must be no longer than 8 sentences.",
    ],
)

uc_accuracy = make_judge(
    name="uc_technical_accuracy",
    instructions="""
Evaluate whether {{ outputs }} is technically accurate as an answer to the
Unity Catalog question in {{ inputs }}.

Score 1-5:
  5 = Precise + caveats   4 = Mostly correct, minor omissions
  3 = Partially correct   2 = Significant inaccuracy or misleading
  1 = Fundamentally wrong / hallucinated APIs

Reason briefly. Return the integer score on the final line.
""",
    model="databricks:/databricks-claude-opus-4-6",
)

PROD_SCORERS = {
    "safety":                 (Safety(),                1.0),    # safety on every request
    "compliance":             (compliance,              0.5),    # 50% sample
    "retrieval_groundedness": (RetrievalGroundedness(), 0.3),    # 30% sample
    "correctness":            (Correctness(),           0.2),    # 20% sample (needs ground truth — only fires when present)
    "uc_technical_accuracy":  (uc_accuracy,             0.2),    # 20% sample (expensive)
}

print("✅ Production scorers + sample rates:")
for name, (_, rate) in PROD_SCORERS.items():
    print(f"  {name:24s}  sample_rate = {rate:.2f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Register the Scorers Against the Endpoint
# MAGIC
# MAGIC `scorer.register(name=...).start(sample_rate=...)` attaches a scorer to the experiment-bound monitoring service. From then on, MLflow runs the judge on the configured fraction of requests asynchronously — production latency stays untouched.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔧 STEP 3 - REGISTER + START SCORERS
# ============================================================================

registrations = {}
for name, (s, rate) in PROD_SCORERS.items():
    try:
        registered = s.register(name=f"capstone_{name}")
        registered.start(sample_rate=rate)
        registrations[name] = registered
        print(f"✅ Registered + started capstone_{name} (sample_rate={rate})")
    except Exception as e:
        print(f"⚠️  capstone_{name} register failed: {e!r}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Simulate 50 Production Requests
# MAGIC
# MAGIC Hammer the endpoint with realistic UC governance questions plus a few off-topic / borderline rows so monitoring sees both clean and noisy traffic.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🎯 STEP 4 - SIMULATE 50 PROD REQUESTS
# ============================================================================

import random, time
from databricks.sdk import WorkspaceClient

oai = WorkspaceClient().serving_endpoints.get_open_ai_client()

BASE_QUESTIONS = [
    "What is the Unity Catalog three-level namespace?",
    "How do I grant SELECT on a table to a group?",
    "How long does Unity Catalog retain lineage?",
    "Explain the difference between row filters and column masks.",
    "How do volumes differ from external tables?",
    "How does Delta Sharing handle revocation?",
    "What replaced model stages in Unity Catalog?",
    "Where do Unity Catalog audit logs live?",
    "Can I edit system.access.audit?",
    "What permissions are needed to read a UC table?",
]
OFF_TOPIC = [
    "What's the weather in Berlin tomorrow?",
    "Recommend a vegetarian recipe with chickpeas.",
]

TOTAL = 50
questions = (BASE_QUESTIONS * 5)[:TOTAL - len(OFF_TOPIC)] + OFF_TOPIC
random.shuffle(questions)

for i, q in enumerate(questions, 1):
    try:
        oai.chat.completions.create(
            model=SERVING_ENDPOINT,
            messages=[{"role": "user", "content": q}],
        )
    except Exception as e:
        # Gateway-blocked / safety-blocked still counts — keep the loop running.
        pass
    if i % 10 == 0:
        print(f"  …{i}/{TOTAL} requests sent")
    time.sleep(0.05)

print(f"\n✅ Sent {TOTAL} simulated production requests.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Verify Judge Feedback Lands on Traces
# MAGIC
# MAGIC Registered scorers run **asynchronously** — give them ~30-60 s and then query the experiment for recent traces. Each `Assessment` carries the judge name and value.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔬 STEP 5 - VERIFY ASSESSMENTS ON TRACES
# ============================================================================

import time

print("⏳ Waiting 60 s for async scorers to finish…")
time.sleep(60)

traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    max_results=20,
    order_by=["timestamp_ms DESC"],
)

print(f"Recent traces: {len(traces)}")
if len(traces):
    display(traces.head(10))

expected = {f"capstone_{n}" for n in PROD_SCORERS}
seen     = set()
for tr in traces.itertuples():
    for a in (getattr(tr, "assessments", []) or []):
        if hasattr(a, "name"):
            seen.add(a.name)

print("\nJudge coverage on recent traces:")
for n in expected:
    mark = "✅" if n in seen else "⏳ (sample rate may not have hit yet)"
    print(f"  {n:30s} {mark}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Inference Table Health Check
# MAGIC
# MAGIC The AI Gateway inference table from 8.1 is the canonical record of every request. Status code mix, gateway-blocked count, and latency P95 are the headline operational numbers.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🩺 STEP 6 - INFERENCE TABLE HEALTH
# ============================================================================

from pyspark.sql import functions as F

last_24h = (
    spark.table(INFERENCE_TABLE)
         .filter(F.col("timestamp_ms") > (F.unix_timestamp() - 86400) * 1000)
)

display(last_24h.agg(
    F.count("*").alias("total_requests"),
    F.sum(F.when(F.col("status_code") == 200, 1).otherwise(0)).alias("passed"),
    F.sum(F.when(F.col("status_code") != 200, 1).otherwise(0)).alias("blocked_or_errored"),
    F.expr("percentile_approx(execution_duration_ms, 0.5)").alias("latency_p50_ms"),
    F.expr("percentile_approx(execution_duration_ms, 0.95)").alias("latency_p95_ms"),
))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Daily Production Quality Roll-Up
# MAGIC
# MAGIC Join the inference table aggregate with the trace-level scorer output to build the dashboard you'd page on. We compute mean / pass-ratio per scorer over the same 24-hour window.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📊 STEP 7 - DAILY QUALITY ROLL-UP
# ============================================================================

import pandas as pd

rows = []
for tr in traces.itertuples():
    for a in (getattr(tr, "assessments", []) or []):
        if hasattr(a, "name") and hasattr(a, "feedback"):
            v = getattr(a.feedback, "value", None)
            rows.append({"name": a.name, "value": v})

if rows:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("name")["value"]
          .agg(
              count="size",
              mean=lambda s: pd.to_numeric(s, errors="coerce").mean(),
              pass_ratio=lambda s: (s == "yes").mean() if s.dtype == object else None,
          )
          .reset_index()
    )
    display(summary)
else:
    print("ℹ️  No trace assessments yet — wait another 30 s and re-run Steps 5+7.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Drift Response Playbook
# MAGIC
# MAGIC When a registered scorer starts flooding alerts, you don't need to redeploy — adjust the schedule on the running registration.
# MAGIC
# MAGIC | Symptom | Action |
# MAGIC | --- | --- |
# MAGIC | Judge becoming a P0 cost line | Lower its sample rate (`reg.update(sample_rate=…)`) |
# MAGIC | Judge consistently mis-firing | `reg.stop()`, fix prompt offline, re-`register().start()` |
# MAGIC | Need a brand-new dimension | Add a new `@scorer`/`Guidelines`, register, leave existing scorers alone |
# MAGIC | Cost spike during incident | Pause non-`Safety` scorers (`reg.stop()`); leave `Safety` at 1.0 |
# MAGIC

# COMMAND ----------

# ============================================================================
# 🛠️ STEP 8 - DRIFT RESPONSE EXAMPLES
# ============================================================================

# Example: lower the cost of the expensive UC accuracy judge in an incident.
if "uc_technical_accuracy" in registrations:
    try:
        registrations["uc_technical_accuracy"].update(sample_rate=0.05)
        print("✅ uc_technical_accuracy sample_rate lowered to 0.05")
    except Exception as e:
        print(f"ℹ️  update() not available in this runtime: {e!r}")

# Example: pause compliance for an hour during a known false-positive spike.
# registrations['compliance'].stop()
# ... fix prompt ...
# registrations['compliance'].start(sample_rate=0.5)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | Five scorers registered against the live endpoint with per-judge sample rates | ✅ |
# MAGIC | 50 simulated production requests sent | ✅ |
# MAGIC | Recent traces show judge `Assessment` rows attached | ✅ |
# MAGIC | Inference table health (status codes + latency) computed | ✅ |
# MAGIC | Daily quality roll-up (mean / pass_ratio per judge) | ✅ |
# MAGIC | Drift playbook — lower / pause / swap a registered scorer without redeploy | ✅ |
# MAGIC
# MAGIC **Next:** *Capstone 8.6* — close the loop with **human feedback** via `mlflow.log_feedback(...)`, compare automated judges to human labels, iterate the judge prompt, and re-run the gate with improved judges.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we built the **Monitoring layer** of the capstone:
# MAGIC
# MAGIC ### 1. Registered Scorers Run Async
# MAGIC - `.register().start(sample_rate=...)` doesn't tax production latency — judges run after the response returns.
# MAGIC - Sample rates are the cost dial: 1.0 for `Safety`, 0.05–0.2 for expensive LLM judges.
# MAGIC
# MAGIC ### 2. Two Tables, One Picture
# MAGIC - The **inference table** captures *what* happened (status, latency, gateway verdict).
# MAGIC - **Trace assessments** capture *how good* it was (judge values + rationales).
# MAGIC - Joined, they're the dashboard you'd page on.
# MAGIC
# MAGIC ### 3. Drift Response Without Redeploy
# MAGIC - Lower / pause / swap a registered scorer in seconds via the registration handle — the deployed model never moves.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Move to **Capstone 8.6** to close the loop: human feedback, agreement analysis, judge iteration, re-gate.
# MAGIC