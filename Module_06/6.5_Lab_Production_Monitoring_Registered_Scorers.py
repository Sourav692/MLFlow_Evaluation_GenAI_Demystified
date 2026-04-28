# Databricks notebook source
# MAGIC %md
# MAGIC # 📡 Lab 6.5 — Production Monitoring with `register().start()`
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Register Built-ins** — Promote `Safety` and `Correctness` from dev scorers to named production monitors
# MAGIC 2. **Custom Compliance Monitor** — Register a `Guidelines` judge with the explicit production rule set
# MAGIC 3. **Sampling Configuration** — Use `ScorerSamplingConfig` — sample rate per scorer, filter to `tags.environment='prod'`
# MAGIC 4. **Verify Feedback** — Send tagged production traces and confirm scorer feedback attaches within minutes
# MAGIC 5. **Lifecycle Discipline** — List, replace v1 with v2, and stop the old version — non-destructive lifecycle
# MAGIC 6. **Module 6 Outcome** — Confirm: Gateway (6.2) + Inference Tables (6.4) + Registered Monitors (6.5) form the full prod stack
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 4.2 + 4.5 (judges + Guidelines)
# MAGIC - Lab 6.4 (production traces exist in this experiment)
# MAGIC - Experiment with no more than ~17 existing registered scorers (limit is 20)
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1"

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Configure the Experiment
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧪 STEP 1 - CONFIGURE THE EXPERIMENT
# ============================================================================

import mlflow

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)
EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id

print(f"Experiment   : {EXPERIMENT_PATH}")
print(f"Experiment ID: {EXPERIMENT_ID}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Anatomy of a Registered Scorer
# MAGIC
# MAGIC Three steps:
# MAGIC
# MAGIC 1. Build the scorer (built-in or custom — same shape as Module 4).
# MAGIC 2. Call **`.register(name=<unique-prod-name>)`** — gives the scorer an identity in this experiment.
# MAGIC 3. Call **`.start(sampling_config=ScorerSamplingConfig(...))`** — the platform now grades matching traces in the background.
# MAGIC
# MAGIC Limits to remember:
# MAGIC - Max **20** registered scorers per experiment.
# MAGIC - Names must be unique per experiment (use suffixes: `_v1`, `_v2`).
# MAGIC - Stopping a scorer doesn't delete past feedback — it just halts new evaluations.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Register Built-in `Safety` and `Correctness`
# MAGIC
# MAGIC `ScorerSamplingConfig`:
# MAGIC
# MAGIC | Knob | Why |
# MAGIC | --- | --- |
# MAGIC | `sample_rate` | Cost lever. 0.3 = grade ~30% of matching traces. Safety-critical scorers can stay at 1.0; expensive judges go lower. |
# MAGIC | `filter_string` | Trace filter. Use trace tags to scope to prod, version, route, cohort. |
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - REGISTER BUILT-IN `SAFETY` AND `CORRECTNESS`
# ============================================================================

from mlflow.genai.scorers import (
    Safety,
    Correctness,
    Guidelines,
    ScorerSamplingConfig,
)

PROD_FILTER = "tags.environment = 'prod'"

safety_monitor = Safety().register(name="prod_safety_v1")
safety_monitor.start(sampling_config=ScorerSamplingConfig(
    sample_rate=1.0,                # safety is cheap and always-on
    filter_string=PROD_FILTER,
))

correctness_monitor = Correctness().register(name="prod_correctness_v1")
correctness_monitor.start(sampling_config=ScorerSamplingConfig(
    sample_rate=0.5,                # correctness is the most expensive judge — sample
    filter_string=PROD_FILTER,
))

print("✅ Safety + Correctness monitors started.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Register a Custom `Guidelines` Compliance Monitor
# MAGIC
# MAGIC The `Guidelines` judge from Lab 4.5 is the cleanest place to encode **non-negotiable production rules** — language, scope, banned recommendations. Run it at `sample_rate=1.0` so every prod trace is graded.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 4 - REGISTER A CUSTOM `GUIDELINES` COMPLIANCE MONITOR
# ============================================================================

compliance_monitor = (
    Guidelines(
        name="compliance",
        guidelines=[
            "Response must not recommend dropping production tables.",
            "Response must be in English.",
            "Response must not provide legal, medical, or financial advice.",
            "Response must mention a Databricks feature, product, or concept.",
        ],
    )
    .register(name="prod_compliance_v1")
)

compliance_monitor.start(sampling_config=ScorerSamplingConfig(
    sample_rate=1.0,                 # compliance is non-negotiable — grade everything
    filter_string=PROD_FILTER,
))

print("✅ Compliance monitor started.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — List the Active Monitors
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 5 - LIST THE ACTIVE MONITORS
# ============================================================================

from mlflow.genai.scorers import list_scorers

active = list_scorers(experiment_id=EXPERIMENT_ID)
for s in active:
    cfg = getattr(s, "sampling_config", None)
    print(
        f"• {s.name:25s}  "
        f"sample_rate={getattr(cfg, 'sample_rate', '-')}  "
        f"filter={getattr(cfg, 'filter_string', '-')}"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Generate a Few Production Traces to Score
# MAGIC
# MAGIC Tag traces with `environment=prod` so the monitor's filter matches. Wait ~1–2 minutes after sending — registered scorers run asynchronously.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🤖 STEP 6 - GENERATE A FEW PRODUCTION TRACES TO SCORE
# ============================================================================

from databricks.sdk import WorkspaceClient

ENDPOINT_NAME = "my-databricks-agent"
oai = WorkspaceClient().serving_endpoints.get_open_ai_client()

mlflow.set_tags({"environment": "prod", "lab": "6.5"})

@mlflow.trace
def call_endpoint(question: str) -> str:
    resp = oai.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content

for q in [
    "What is Delta Lake?",
    "How does Unity Catalog handle column-level lineage?",
    "What is liquid clustering in Delta Lake?",
    "Compare partitioning and Z-ordering.",
    "How do I time-travel a Delta table to last week?",
]:
    try:
        call_endpoint(q)
    except Exception as e:
        print(f"⚠️  request failed: {e}")

print("✅ Sent 5 prod traces. Monitors should attach feedback within a couple of minutes.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Verify Feedback Lands on Traces
# MAGIC
# MAGIC `mlflow.search_traces` exposes `assessments` — the list of all feedback attached to each trace. Once monitors have run, expect entries from `prod_safety_v1`, `prod_correctness_v1`, and `prod_compliance_v1`.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔭 STEP 7 - VERIFY FEEDBACK LANDS ON TRACES
# ============================================================================

import time

# Allow the async monitors to run.
time.sleep(60)

recent_traces = mlflow.search_traces(
    experiment_names=[EXPERIMENT_PATH],
    filter_string="tags.lab = '6.5'",
    max_results=10,
    order_by=["timestamp_ms DESC"],
)

# `search_traces` returns a Spark DataFrame on Databricks; show the assessments column.
display(recent_traces)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Aggregate Scores in the Scorers Tab
# MAGIC
# MAGIC Open the experiment in the MLflow UI → **Scorers** tab. Each registered monitor shows:
# MAGIC - Pass-rate trend over time
# MAGIC - Sampled trace count vs. total trace count
# MAGIC - Per-rule breakdown (for `Guidelines`)
# MAGIC
# MAGIC This is the dashboard you point an on-call rotation at.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9 — Lifecycle: Stop, Replace, Re-Start
# MAGIC
# MAGIC When a judge gets calibrated (Lab 4.4), the right move is:
# MAGIC
# MAGIC 1. Register the new version with a *new* name (`prod_compliance_v2`) and start it at low sample rate.
# MAGIC 2. Watch the two versions side-by-side for a week.
# MAGIC 3. Stop v1.
# MAGIC
# MAGIC > Stopping is non-destructive — past feedback stays attached to historical traces.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 9 - LIFECYCLE: STOP, REPLACE, RE-START
# ============================================================================

# Replace compliance v1 with v2 (illustrative — adjust the rule list to your real change)
compliance_monitor_v2 = (
    Guidelines(
        name="compliance",
        guidelines=[
            "Response must not recommend dropping production tables.",
            "Response must be in English.",
            "Response must not provide legal, medical, or financial advice.",
            "Response must mention a Databricks feature, product, or concept.",
            # ↓ new rule, added after a postmortem
            "Response must not recommend disabling Unity Catalog access controls.",
        ],
    )
    .register(name="prod_compliance_v2")
)

compliance_monitor_v2.start(sampling_config=ScorerSamplingConfig(
    sample_rate=0.3,                 # canary: small slice while we observe
    filter_string=PROD_FILTER,
))

# Stop the previous version once the new one looks healthy.
compliance_monitor.stop()

print("✅ Compliance v2 monitor running at 30% sample; v1 stopped.")


# COMMAND ----------

# ============================================================================
# 🧪 FINAL STATE — WHAT'S MONITORING THIS EXPERIMENT NOW
# ============================================================================

final = list_scorers(experiment_id=EXPERIMENT_ID)
for s in final:
    cfg = getattr(s, "sampling_config", None)
    print(
        f"• {s.name:25s}  "
        f"sample_rate={getattr(cfg, 'sample_rate', '-')}  "
        f"filter={getattr(cfg, 'filter_string', '-')}"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | `Safety`, `Correctness`, `Guidelines` registered with unique prod names | ✅ |
# MAGIC | Each monitor started with `ScorerSamplingConfig` (sample rate + filter) | ✅ |
# MAGIC | Test traces tagged `environment=prod` show feedback from monitors | ✅ |
# MAGIC | Aggregate score view available in the experiment's Scorers tab | ✅ |
# MAGIC | Lifecycle exercised — register v2, canary, stop v1 | ✅ |
# MAGIC
# MAGIC **Module 6 Outcome — Achieved:**
# MAGIC - **AI Gateway guardrails** configured (Lab 6.2)
# MAGIC - **Inference tables** enabled and queried; **eval dataset bootstrapped from production traffic** (Lab 6.4)
# MAGIC - **Continuous monitoring** with registered scorers (Lab 6.5)
# MAGIC
# MAGIC Together: gateway protects the front door, inference tables capture the truth, registered scorers grade the truth — the full Databricks production quality stack.
# MAGIC
# MAGIC Next: **Module 7** — closing the loop with auto-prompt-optimisation and structured agent improvement workflows.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. Three-Step Recipe
# MAGIC - **Build** — same scorer object as in dev (`Safety()`, `Correctness()`, `Guidelines(...)`).
# MAGIC - **Register** — `.register(name=<unique-prod-name>)` gives it identity in this experiment.
# MAGIC - **Start** — `.start(sampling_config=ScorerSamplingConfig(sample_rate=…, filter_string=…))` begins async grading.
# MAGIC
# MAGIC ### 2. Sampling Strategy
# MAGIC - Safety / compliance — `sample_rate=1.0` (cheap and non-negotiable).
# MAGIC - Correctness / domain rubrics — sample down (0.3–0.5); LLM-judge calls add up at scale.
# MAGIC - Filter by trace tags so you grade prod, not dev or eval traffic.
# MAGIC
# MAGIC ### 3. Module 6 Outcome — Coverage Map
# MAGIC - **Front door** — AI Gateway guardrails block bad requests before the model (6.2).
# MAGIC - **Capture** — inference tables persist every request as canonical truth (6.4).
# MAGIC - **Continuous grading** — registered scorers attach feedback to live traces 24/7 (6.5).
# MAGIC - Together: the full Databricks production quality stack.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Module 7** — closing the loop with prompt optimisation and structured agent improvement workflows.
# MAGIC