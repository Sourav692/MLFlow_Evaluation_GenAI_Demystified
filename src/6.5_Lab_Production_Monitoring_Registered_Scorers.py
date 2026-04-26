# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 6.5 — Production Monitoring with `register().start()`
# MAGIC
# MAGIC **Goal:** Promote the dev-time scorers from Modules 4–5 into **continuous production monitors** that grade live traces in the background — sampled, filtered, and visible as feedback in the MLflow UI.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. **Registered** `Safety`, `Correctness`, and a custom `Guidelines` scorer with unique production names
# MAGIC 2. **Started** each monitor with a `ScorerSamplingConfig` — sample rate + filter for production-only traces
# MAGIC 3. Sent a handful of test traces and **verified scorer feedback appears** on each within minutes
# MAGIC 4. Inspected **aggregate scores** in the experiment's Scorers tab — trend view over time
# MAGIC 5. Practiced the **lifecycle**: list registered scorers, stop one, replace it with a new version
# MAGIC
# MAGIC > **Why this lab matters:** Offline eval catches regressions on a snapshot. **Continuous scorers catch them on live traffic**, in minutes, with the same judges you've already calibrated. This is the closing of the production-quality loop — gateway in front, registered scorers behind.
# MAGIC
# MAGIC > **Prereq:** Lab 4.2 (judges), Lab 4.5 (`Guidelines`), Lab 6.4 (you've sent traffic to the endpoint and have traces in this experiment).

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure the Experiment

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Register Built-in `Safety` and `Correctness`
# MAGIC
# MAGIC `ScorerSamplingConfig`:
# MAGIC
# MAGIC | Knob | Why |
# MAGIC | --- | --- |
# MAGIC | `sample_rate` | Cost lever. 0.3 = grade ~30% of matching traces. Safety-critical scorers can stay at 1.0; expensive judges go lower. |
# MAGIC | `filter_string` | Trace filter. Use trace tags to scope to prod, version, route, cohort. |

# COMMAND ----------

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
# MAGIC ## Step 4 — Register a Custom `Guidelines` Compliance Monitor
# MAGIC
# MAGIC The `Guidelines` judge from Lab 4.5 is the cleanest place to encode **non-negotiable production rules** — language, scope, banned recommendations. Run it at `sample_rate=1.0` so every prod trace is graded.

# COMMAND ----------

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
# MAGIC ## Step 5 — List the Active Monitors

# COMMAND ----------

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
# MAGIC ## Step 6 — Generate a Few Production Traces to Score
# MAGIC
# MAGIC Tag traces with `environment=prod` so the monitor's filter matches. Wait ~1–2 minutes after sending — registered scorers run asynchronously.

# COMMAND ----------

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
# MAGIC ## Step 7 — Verify Feedback Lands on Traces
# MAGIC
# MAGIC `mlflow.search_traces` exposes `assessments` — the list of all feedback attached to each trace. Once monitors have run, expect entries from `prod_safety_v1`, `prod_correctness_v1`, and `prod_compliance_v1`.

# COMMAND ----------

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
# MAGIC ## Step 8 — Aggregate Scores in the Scorers Tab
# MAGIC
# MAGIC Open the experiment in the MLflow UI → **Scorers** tab. Each registered monitor shows:
# MAGIC - Pass-rate trend over time
# MAGIC - Sampled trace count vs. total trace count
# MAGIC - Per-rule breakdown (for `Guidelines`)
# MAGIC
# MAGIC This is the dashboard you point an on-call rotation at.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Lifecycle: Stop, Replace, Re-Start
# MAGIC
# MAGIC When a judge gets calibrated (Lab 4.4), the right move is:
# MAGIC
# MAGIC 1. Register the new version with a *new* name (`prod_compliance_v2`) and start it at low sample rate.
# MAGIC 2. Watch the two versions side-by-side for a week.
# MAGIC 3. Stop v1.
# MAGIC
# MAGIC > Stopping is non-destructive — past feedback stays attached to historical traces.

# COMMAND ----------

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

# DBTITLE 1,Final state — what's monitoring this experiment now
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
