# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 7.2 — Collect & Store User Feedback via API
# MAGIC
# MAGIC **Goal:** Wire up the path from a **user thumbs-up / thumbs-down in your UI** to a typed **`Feedback` assessment attached to the right MLflow trace** — and show that human feedback sits side-by-side with automated judge scores.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Sent **5 traced requests**, each tagged with a unique `client_request_id` (the join key your UI controls)
# MAGIC 2. Resolved a `client_request_id` → MLflow `trace_id` via **`mlflow.search_traces(filter_string=…)`**
# MAGIC 3. Logged a human assessment with **`mlflow.log_feedback(trace_id=…, name="user_helpful", value=True, source=AssessmentSource(source_type="HUMAN", source_id="user_123"))`**
# MAGIC 4. Logged a free-text rationale and a numeric satisfaction score on the same trace
# MAGIC 5. Compared **aggregate human feedback vs. automated judge scores** for the same set of traces
# MAGIC
# MAGIC > **Why this lab matters:** Human feedback is the **ground truth** that calibrates everything else (Lab 7.3). The contract is small: your UI emits `client_request_id`, the agent sets that as a trace tag, and the feedback endpoint uses it to find the trace. Once that loop is closed, every product event becomes labelled training data for free.
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 (workspace + experiment), Lab 4.2 (you've used built-in judges), Lab 6.5 (registered scorers — useful for the comparison view).

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure Experiment

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

print(f"Experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — The Feedback Loop in One Picture
# MAGIC
# MAGIC ```
# MAGIC  ┌───────────────┐    client_request_id     ┌──────────────┐
# MAGIC  │  Your UI      │ ─────────────────────▶  │   Agent      │
# MAGIC  │ (ChatBox)     │                         │  @mlflow.trace│
# MAGIC  └─────┬─────────┘                         └──────┬───────┘
# MAGIC        │  thumbs-up                                │ logs trace
# MAGIC        │  client_request_id="abc"                  ▼
# MAGIC        │                                  ┌──────────────────┐
# MAGIC        ▼                                  │ MLflow trace store│
# MAGIC  /api/feedback ───── search_traces ─────▶  └──────────────────┘
# MAGIC        │                                          │
# MAGIC        └──── log_feedback(trace_id, …) ──────────▶│
# MAGIC ```
# MAGIC
# MAGIC The contract is one piece of data: **`client_request_id`**. Anything your UI can persist next to a feedback button is a valid choice — request UUID, conversation turn ID, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build a Tiny Agent That Tags Each Trace with `client_request_id`
# MAGIC
# MAGIC The pattern: take `client_request_id` as an argument, set it as a **trace tag** in the very first line of the function. That makes it queryable from `search_traces` later.

# COMMAND ----------

import uuid
from databricks.sdk import WorkspaceClient

oai = WorkspaceClient().serving_endpoints.get_open_ai_client()

@mlflow.trace
def chat_agent(question: str, client_request_id: str) -> str:
    # Tag THIS trace with the UI-provided ID so the feedback endpoint can find it later.
    mlflow.update_current_trace(tags={"client_request_id": client_request_id})

    resp = oai.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": "You are a Databricks expert. Answer concisely."},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Send 5 Traced Requests with Unique `client_request_id`s

# COMMAND ----------

QUESTIONS = [
    "What is Delta Lake?",
    "Explain Z-ordering.",
    "How does Unity Catalog handle column-level lineage?",
    "What is the default VACUUM retention?",
    "Compare partitioning and Z-ordering.",
]

requests_log = []  # what your UI would persist alongside the chat history
for q in QUESTIONS:
    crid = f"req-{uuid.uuid4().hex[:8]}"
    answer = chat_agent(q, client_request_id=crid)
    requests_log.append({"client_request_id": crid, "question": q, "answer": answer})

for r in requests_log:
    print(f"{r['client_request_id']}  ▸  {r['question']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Resolve `client_request_id` → `trace_id`
# MAGIC
# MAGIC This is the lookup your `/api/feedback` endpoint runs on each user click. It's exactly **one** `search_traces` call.

# COMMAND ----------

def trace_id_for(client_request_id: str) -> str:
    df = mlflow.search_traces(
        experiment_names=[EXPERIMENT_PATH],
        filter_string=f"tags.client_request_id = '{client_request_id}'",
        max_results=1,
    )
    rows = df.collect() if hasattr(df, "collect") else df.to_dict("records")
    if not rows:
        raise LookupError(f"No trace found for client_request_id={client_request_id}")
    row = rows[0]
    return row["trace_id"] if isinstance(row, dict) else row.trace_id

# Sanity check on the first request
sample_crid = requests_log[0]["client_request_id"]
sample_trace = trace_id_for(sample_crid)
print(f"client_request_id={sample_crid} → trace_id={sample_trace}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Simulate a User Thumbs-Up: `mlflow.log_feedback(...)`
# MAGIC
# MAGIC `log_feedback` attaches a typed **`Feedback`** assessment to the trace. Three things matter:
# MAGIC
# MAGIC | Field | Why |
# MAGIC | --- | --- |
# MAGIC | `name`      | column name in eval results — keep it stable across product versions |
# MAGIC | `value`     | the rating itself: bool, str, or numeric |
# MAGIC | `source`    | `AssessmentSource(source_type="HUMAN", source_id="user_123")` — provenance for filtering & calibration |

# COMMAND ----------

from mlflow.entities import AssessmentSource

# Simulate two thumbs-ups, two thumbs-downs, one neutral
SIMULATED_FEEDBACK = [
    {"crid": requests_log[0]["client_request_id"], "user": "user_123", "value": True,  "rationale": "Concise, accurate."},
    {"crid": requests_log[1]["client_request_id"], "user": "user_456", "value": True,  "rationale": "Z-order explanation matched my mental model."},
    {"crid": requests_log[2]["client_request_id"], "user": "user_789", "value": False, "rationale": "Missed the column-level lineage detail."},
    {"crid": requests_log[3]["client_request_id"], "user": "user_456", "value": False, "rationale": "Said retention is 30 days, actual default is 7."},
    {"crid": requests_log[4]["client_request_id"], "user": "user_321", "value": True,  "rationale": "Good comparison table."},
]

for fb in SIMULATED_FEEDBACK:
    tid = trace_id_for(fb["crid"])
    mlflow.log_feedback(
        trace_id=tid,
        name="user_helpful",
        value=fb["value"],
        rationale=fb["rationale"],
        source=AssessmentSource(source_type="HUMAN", source_id=fb["user"]),
    )

print("✅ 5 user feedback rows logged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Log a Numeric Satisfaction Score Alongside
# MAGIC
# MAGIC Most product UIs collect more than thumbs — a 1-5 star rating, a topic tag, a free-text comment. Each one is a separate `log_feedback` call with its own `name`. They all attach to the same trace.

# COMMAND ----------

SATISFACTION = {
    requests_log[0]["client_request_id"]: 5,
    requests_log[1]["client_request_id"]: 4,
    requests_log[2]["client_request_id"]: 2,
    requests_log[3]["client_request_id"]: 1,
    requests_log[4]["client_request_id"]: 5,
}

for crid, score in SATISFACTION.items():
    mlflow.log_feedback(
        trace_id=trace_id_for(crid),
        name="satisfaction_1_5",
        value=score,
        source=AssessmentSource(source_type="HUMAN", source_id="aggregate"),
    )

print("✅ Satisfaction scores logged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Pull All Feedback for the Test Cohort
# MAGIC
# MAGIC Each trace's `assessments` column contains every assessment ever attached — humans + automated judges + registered scorers. Filter by `source.source_type` to slice human vs. machine.

# COMMAND ----------

crids = [r["client_request_id"] for r in requests_log]
crid_filter = " OR ".join(f"tags.client_request_id = '{c}'" for c in crids)

cohort = mlflow.search_traces(
    experiment_names=[EXPERIMENT_PATH],
    filter_string=crid_filter,
    max_results=20,
)
display(cohort)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Compare Human Feedback vs. Automated Judges
# MAGIC
# MAGIC Run an automated judge (`Correctness`) over the same questions, then put both score columns side-by-side. The point isn't "humans are right and judges are wrong" — it's that you can now **measure agreement** (Lab 7.3 turns this into a calibration loop).

# COMMAND ----------

import pandas as pd
from mlflow.genai.scorers import Correctness, Safety

# Build a small eval dataframe matching the 5 questions, with empty expected_facts.
# Correctness will grade against `expected_response` if provided; without it, it
# uses LLM-as-judge to verify the answer is plausible. We pre-fill expected_facts
# from the inference table semantics for clarity.

EXPECTED_FACTS = {
    "What is Delta Lake?":                                 ["ACID transactions", "schema enforcement", "time travel"],
    "Explain Z-ordering.":                                 ["co-locates related data", "improves data skipping"],
    "How does Unity Catalog handle column-level lineage?": ["column-level lineage", "Unity Catalog"],
    "What is the default VACUUM retention?":               ["7 days"],
    "Compare partitioning and Z-ordering.":                ["partitioning", "Z-ordering"],
}

eval_df = pd.DataFrame([
    {"inputs": {"question": q}, "expectations": {"expected_facts": EXPECTED_FACTS[q]}}
    for q in QUESTIONS
])

@mlflow.trace
def predict_for_eval(question: str) -> str:
    return chat_agent(question, client_request_id=f"eval-{uuid.uuid4().hex[:8]}")

results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=predict_for_eval,
    scorers=[Correctness(), Safety()],
    model_id="models:/feedback-cohort/v1",
)

display(results.tables["eval_results"].select(
    "inputs",
    "outputs",
    "`correctness/v1/value`",
    "`safety/v1/value`",
))

# COMMAND ----------

# DBTITLE 1,Side-by-side: human feedback vs. automated judges
import pandas as pd

human_df = pd.DataFrame([
    {"question": next(r["question"] for r in requests_log if r["client_request_id"] == fb["crid"]),
     "user_helpful":  fb["value"],
     "satisfaction":  SATISFACTION[fb["crid"]]}
    for fb in SIMULATED_FEEDBACK
])

print("Human feedback aggregate:")
print(f"  thumbs_up_rate  = {human_df['user_helpful'].mean():.2f}")
print(f"  mean_satisfaction = {human_df['satisfaction'].mean():.2f}/5")

judge_df = (results.tables["eval_results"]
                .selectExpr(
                    "AVG(CASE WHEN `correctness/v1/value` = 'yes' THEN 1.0 ELSE 0.0 END) AS correctness_pass",
                    "AVG(CASE WHEN `safety/v1/value`      = 'yes' THEN 1.0 ELSE 0.0 END) AS safety_pass",
                ).toPandas())
print("\nAutomated judge aggregate:")
print(judge_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — What Your `/api/feedback` Endpoint Looks Like
# MAGIC
# MAGIC Wrap Steps 5 + 6 into a function and you have the production endpoint. Hosted as a Databricks App, FastAPI service, or serverless function — same code.

# COMMAND ----------

def submit_user_feedback(client_request_id: str,
                         user_id: str,
                         helpful: bool,
                         rationale: str | None = None,
                         satisfaction: int | None = None) -> dict:
    """Production-shaped feedback handler — call from your UI's API."""
    tid = trace_id_for(client_request_id)
    mlflow.log_feedback(
        trace_id=tid,
        name="user_helpful",
        value=helpful,
        rationale=rationale,
        source=AssessmentSource(source_type="HUMAN", source_id=user_id),
    )
    if satisfaction is not None:
        mlflow.log_feedback(
            trace_id=tid,
            name="satisfaction_1_5",
            value=satisfaction,
            source=AssessmentSource(source_type="HUMAN", source_id=user_id),
        )
    return {"trace_id": tid, "ok": True}

# Example UI call:
print(submit_user_feedback(
    client_request_id=requests_log[0]["client_request_id"],
    user_id="user_999",
    helpful=True,
    rationale="Used this answer in my onboarding doc.",
    satisfaction=5,
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | 5 requests sent, each tagged with a unique `client_request_id` | ✅ |
# MAGIC | `search_traces(filter_string=…)` resolves `client_request_id` → `trace_id` | ✅ |
# MAGIC | `log_feedback` attached typed human assessments (helpful + satisfaction) | ✅ |
# MAGIC | Same traces also have automated judge scores from `Correctness` / `Safety` | ✅ |
# MAGIC | Side-by-side aggregate of human vs. machine scores produced | ✅ |
# MAGIC | Production-shaped `submit_user_feedback()` endpoint demonstrated | ✅ |
# MAGIC
# MAGIC Next: **Lab 7.3** — turn that human feedback into a **judge calibration loop** — measure agreement, refine the rubric, re-evaluate.
# MAGIC
# MAGIC After that: **Lab 7.4** — wire the calibrated judges into a **pre-deployment quality gate** in Databricks Workflows.
