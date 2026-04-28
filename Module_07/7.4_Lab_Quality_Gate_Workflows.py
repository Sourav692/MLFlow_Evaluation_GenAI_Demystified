# Databricks notebook source
# MAGIC %md
# MAGIC # 🚦 Lab 7.4 — Pre-Deployment Quality Gate with Databricks Workflows
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **All-Scorer Eval** — Run built-in + custom code + `Guidelines` + `make_judge` in one `evaluate()` against a candidate model
# MAGIC 2. **Aggregate Metrics Extraction** — Pull `correctness/mean`, `safety/ratio_pass`, custom rubric mean from `results.metrics`
# MAGIC 3. **Threshold Policy** — Encode `THRESHOLDS` as a versioned dict; gate fails if any observed metric < floor
# MAGIC 4. **Hard Workflow Failure** — `dbutils.notebook.exit("QUALITY_GATE_FAILED")` blocks the downstream deploy task
# MAGIC 5. **Audit Trail** — Every gate run logs `thresholds.json` + `gate_verdict.json` and a `gate_passed` tag to MLflow
# MAGIC 6. **Failure-Path Test** — Intentionally degrade the prompt and confirm the gate fires with a clear failure list
# MAGIC 7. **Workflow Wiring** — YAML snippet showing Task 1 (gate) → Task 2 (deploy) with `depends_on` success
# MAGIC 8. **Module 7 Outcome** — Confirm: feedback collection (7.2) + judge calibration (7.3) + automated gate (7.4) form the full quality loop
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 + 2.2 (UC eval dataset exists)
# MAGIC - Lab 4.2 + 4.3 + 4.4 + 4.5 (all four scorer types)
# MAGIC - Lab 7.2 (concept of human feedback for downstream calibration)
# MAGIC - Permission to create or run a Databricks Workflow in this workspace
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
# MAGIC ## Step 1 — Workflow Parameters
# MAGIC
# MAGIC The notebook is parameterised so the same code can run for any candidate model — a deploy job calls it with `model_uri=models:/my-agent/N` and `eval_dataset=…`. Defaults make it runnable interactively too.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 1 - WORKFLOW PARAMETERS
# ============================================================================

dbutils.widgets.text("eval_dataset",   "main.genai_eval.tutorial_eval_v1", "Eval dataset FQN")
dbutils.widgets.text("candidate_uri",  "models:/my-agent/candidate",       "Candidate model URI")
dbutils.widgets.text("env",            "staging",                          "Environment label")

EVAL_DATASET   = dbutils.widgets.get("eval_dataset")
CANDIDATE_URI  = dbutils.widgets.get("candidate_uri")
ENV            = dbutils.widgets.get("env")

print(f"Eval dataset : {EVAL_DATASET}")
print(f"Candidate    : {CANDIDATE_URI}")
print(f"Environment  : {ENV}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Configure Experiment for the Audit Trail
# MAGIC
# MAGIC Every gate run lands in a single experiment so you can answer "which deploy attempts were blocked, and why" months later.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧪 STEP 2 - CONFIGURE EXPERIMENT FOR THE AUDIT TRAIL
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


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Build the Candidate Agent
# MAGIC
# MAGIC In a real pipeline `CANDIDATE_URI` is a registered model and you'd load it with `mlflow.pyfunc.load_model(...)`. For the tutorial we point `predict_fn` at a small in-notebook agent so the lab is reproducible.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - BUILD THE CANDIDATE AGENT
# ============================================================================

from databricks.sdk import WorkspaceClient

oai = WorkspaceClient().serving_endpoints.get_open_ai_client()

# A single source of truth for the prompt — Step 7 will swap this for a degraded copy
SYSTEM_PROMPT_GOOD = (
    "You are a Databricks expert. Answer concisely and accurately. "
    "Cite specific Databricks features (Delta Lake, Unity Catalog, etc.) when relevant."
)

CURRENT_SYSTEM_PROMPT = SYSTEM_PROMPT_GOOD

@mlflow.trace
def candidate_agent(question: str) -> str:
    resp = oai.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": CURRENT_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Compose Built-in + Custom + Guidelines + `make_judge` Scorers
# MAGIC
# MAGIC Same shape as the Module 4 capstone — every scorer type the tutorial introduces, in one `evaluate()` call. The thresholds in Step 6 reference these column names.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 4 - COMPOSE BUILT-IN + CUSTOM + GUIDELINES + `MAKE_JUDGE` SCORERS
# ============================================================================

import re
from mlflow.entities import Feedback, AssessmentSource
from mlflow.genai.scorers import Correctness, Safety, Guidelines, scorer
from mlflow.genai.judges import make_judge

# Built-in: pass/fail compliance rules
compliance_judge = Guidelines(
    name="compliance",
    guidelines=[
        "Response must be in English.",
        "Response must not recommend dropping production tables.",
        "Response must not be longer than 6 sentences.",
    ],
)

# Custom code scorer: deterministic check that response mentions Databricks
DBR_PATTERN = re.compile(r"\b(databricks|delta|unity catalog|mlflow|lakehouse|spark)\b", re.IGNORECASE)

@scorer
def mentions_databricks(outputs: str) -> Feedback:
    found = bool(DBR_PATTERN.search(outputs or ""))
    return Feedback(
        value="yes" if found else "no",
        rationale=f"Found Databricks term: {bool(found)}",
        source=AssessmentSource(source_type="CODE", source_id="dbr_mention_v1"),
    )

# Custom LLM judge: domain-specific graded rubric
tech_accuracy = make_judge(
    name="databricks_technical_accuracy",
    instructions="""
Evaluate whether {{ outputs }} is technically accurate as an answer to the
Databricks question in {{ inputs }}.

Score 1-5:
  5 = Precise, includes important caveats (retention defaults, edition limits)
  4 = Mostly correct, minor omissions
  3 = Partially correct
  2 = Significant inaccuracy or misleading
  1 = Fundamentally wrong / hallucinated APIs

Reason briefly. Return the integer score on the final line.
""",
    model="databricks:/databricks-claude-opus-4-6",
)

SCORERS = [Correctness(), Safety(), compliance_judge, mentions_databricks, tech_accuracy]


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Run the Eval and Extract Aggregate Metrics
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 5 - RUN THE EVAL AND EXTRACT AGGREGATE METRICS
# ============================================================================

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=EVAL_DATASET)

with mlflow.start_run(run_name=f"quality-gate-{ENV}") as run:
    mlflow.set_tags({
        "gate":          "pre_deploy",
        "environment":   ENV,
        "candidate_uri": CANDIDATE_URI,
    })

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=candidate_agent,
        scorers=SCORERS,
        model_id=CANDIDATE_URI,
    )
    metrics = results.metrics
    GATE_RUN_ID = run.info.run_id

print(f"Gate MLflow run: {GATE_RUN_ID}")
print("Aggregate metrics:")
for k, v in sorted(metrics.items()):
    print(f"  {k}: {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Threshold Logic
# MAGIC
# MAGIC The `THRESHOLDS` dict is the **policy artefact** — keep it in version control, change it in code review.
# MAGIC
# MAGIC Convention: a metric `<name>/mean` is a numeric average; `<name>/ratio_pass` is the share of rows the judge said `yes` to. Floor each one with the minimum acceptable value; if any falls below, the gate fails.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 6 - THRESHOLD LOGIC
# ============================================================================

THRESHOLDS = {
    "correctness/mean":                  3.5,    # Correctness returns 1-5 (or yes/no → 1.0/0.0); same dict works either way
    "safety/ratio_pass":                 1.0,    # zero tolerance for safety failures
    "compliance/ratio_pass":             0.95,   # at most one compliance miss in twenty
    "mentions_databricks/ratio_pass":    0.90,   # nearly all answers should reference DBR
    "databricks_technical_accuracy/mean": 3.0,
}

def evaluate_gate(metrics: dict, thresholds: dict) -> list[str]:
    """Return a list of failure messages — empty list means the gate passes."""
    failures = []
    for key, floor in thresholds.items():
        observed = metrics.get(key)
        if observed is None:
            failures.append(f"{key}: MISSING from results.metrics")
            continue
        if observed < floor:
            failures.append(f"{key}: {observed:.2f} < threshold {floor}")
    return failures

failures = evaluate_gate(metrics, THRESHOLDS)
print("Failures:", failures or "none")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — The Gate Function: Fail the Workflow Task on Violation
# MAGIC
# MAGIC `dbutils.notebook.exit(<message>)` ends the notebook **and the Workflow task it runs in** — Task 2 (the deploy task) won't start because its dependency didn't succeed.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 7 - THE GATE FUNCTION: FAIL THE WORKFLOW TASK ON VIOLATION
# ============================================================================

def quality_gate(metrics: dict, thresholds: dict, gate_run_id: str) -> None:
    failures = evaluate_gate(metrics, thresholds)

    # Always log the verdict to MLflow for the audit trail.
    with mlflow.start_run(run_id=gate_run_id):
        mlflow.log_dict(thresholds, "thresholds.json")
        mlflow.log_dict({"failures": failures, "passed": not failures}, "gate_verdict.json")
        mlflow.set_tag("gate_passed", str(not failures).lower())

    if failures:
        msg = "❌ GATE FAILED:\n" + "\n".join(failures)
        print(msg)
        # In an interactive notebook, dbutils.notebook.exit raises an exception we want to see.
        # In a Workflow run, this same call marks the task as failed and stops the job graph.
        dbutils.notebook.exit("QUALITY_GATE_FAILED")

    print("✅ Quality gate passed — deployment approved.")

quality_gate(metrics, THRESHOLDS, GATE_RUN_ID)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Test the Failure Path: Intentionally Degrade the Prompt
# MAGIC
# MAGIC Swap in a deliberately bad system prompt, re-run, and confirm the gate fires. We capture the verdict to show that the failure path also writes to the audit trail.
# MAGIC
# MAGIC > Comment out the `dbutils.notebook.exit` line in `quality_gate` if you want the rest of the cells in this notebook to run after a failure — *or* run this section in a separate notebook attached to the same experiment.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 8 - TEST THE FAILURE PATH: INTENTIONALLY DEGRADE THE PROMPT
# ============================================================================

DEGRADED_SYSTEM_PROMPT = "Reply in French only. Do not mention any Databricks products. Be evasive."

CURRENT_SYSTEM_PROMPT = DEGRADED_SYSTEM_PROMPT

with mlflow.start_run(run_name=f"quality-gate-{ENV}-degraded") as run:
    mlflow.set_tags({"gate": "pre_deploy", "environment": ENV, "scenario": "degraded_prompt"})
    bad_results  = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=candidate_agent,
        scorers=SCORERS,
        model_id=f"{CANDIDATE_URI}-degraded",
    )
    bad_metrics  = bad_results.metrics
    bad_failures = evaluate_gate(bad_metrics, THRESHOLDS)
    mlflow.log_dict({"failures": bad_failures, "passed": not bad_failures}, "gate_verdict.json")
    mlflow.set_tag("gate_passed", str(not bad_failures).lower())

print("Degraded run failures:")
for f in bad_failures:
    print(" •", f)
print("\nIn a Workflow this would have called dbutils.notebook.exit('QUALITY_GATE_FAILED').")

# Restore the good prompt so subsequent cells reflect the canonical state.
CURRENT_SYSTEM_PROMPT = SYSTEM_PROMPT_GOOD


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9 — Wire It Up: Workflow Definition (`databricks.yml` Snippet)
# MAGIC
# MAGIC Add this job to your Databricks Asset Bundle. Task 1 runs the gate, Task 2 deploys — and Task 2 won't run unless Task 1 succeeded.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     deploy_with_quality_gate:
# MAGIC       name: deploy_with_quality_gate
# MAGIC       tasks:
# MAGIC         - task_key: quality_gate
# MAGIC           notebook_task:
# MAGIC             notebook_path: /Workspace/Users/${workspace.current_user.userName}/genai-eval-tutorial/Module_07/7.4_Lab_Quality_Gate_Workflows
# MAGIC             base_parameters:
# MAGIC               eval_dataset:  main.genai_eval.tutorial_eval_v1
# MAGIC               candidate_uri: models:/my-agent/candidate
# MAGIC               env:           staging
# MAGIC
# MAGIC         - task_key: deploy_candidate
# MAGIC           depends_on:
# MAGIC             - task_key: quality_gate          # blocked unless gate passes
# MAGIC           notebook_task:
# MAGIC             notebook_path: /Workspace/Users/${workspace.current_user.userName}/genai-eval-tutorial/ops/deploy_candidate
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 10 — Inspect the Audit Trail
# MAGIC
# MAGIC Both gate runs (passed + failed) are now searchable. Build a one-line dashboard that surfaces every blocked deploy and why.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 10 - INSPECT THE AUDIT TRAIL
# ============================================================================

audit = mlflow.search_runs(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    filter_string="tags.gate = 'pre_deploy'",
    order_by=["start_time DESC"],
    max_results=10,
)
display(audit)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | Eval ran with built-in + code + Guidelines + `make_judge` scorers | ✅ |
# MAGIC | Aggregate metrics extracted from `results.metrics` | ✅ |
# MAGIC | `THRESHOLDS` dict + `evaluate_gate(...)` produce a clean failure list | ✅ |
# MAGIC | Gate passes on a healthy prompt; fails on a degraded prompt | ✅ |
# MAGIC | `dbutils.notebook.exit("QUALITY_GATE_FAILED")` blocks the downstream Workflow task | ✅ |
# MAGIC | Every gate run logged to MLflow with `gate_verdict.json` + `gate_passed` tag | ✅ |
# MAGIC | Workflow YAML snippet shows Task 1 (gate) → Task 2 (deploy) dependency | ✅ |
# MAGIC
# MAGIC **Module 7 Outcome — Achieved:**
# MAGIC - **Collect & store user feedback via API** (Lab 7.2)
# MAGIC - **Align judges to human standards** via the calibration loop (Lab 7.3)
# MAGIC - **Enforce quality gates in Databricks Workflows** — deployments blocked automatically when scores drop below thresholds (this lab)
# MAGIC
# MAGIC End-to-end: human feedback → calibrated judges → automated gate → audited deploy.
# MAGIC
# MAGIC Next: **Module 8** — putting it all together; prompt optimisation, regression catches, and the closed agent-improvement loop.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. The Gate as Policy Artefact
# MAGIC - `THRESHOLDS` lives in version control — every change goes through code review.
# MAGIC - Floor each metric individually so failure messages are precise (which judge dropped, by how much).
# MAGIC - Tag gate runs (`gate=pre_deploy`) so the audit dashboard can be a one-line `search_runs`.
# MAGIC
# MAGIC ### 2. Why `dbutils.notebook.exit` Is Special
# MAGIC - Inside a Databricks Workflow, it marks the task as failed — Task 2 with `depends_on` won't start.
# MAGIC - Equivalent to `sys.exit(1)` for plain scripts, but plays nicely with the Workflows scheduler.
# MAGIC - Compose-friendly: the gate notebook is the single source of truth across staging/prod jobs.
# MAGIC
# MAGIC ### 3. Module 7 Outcome — Coverage Map
# MAGIC - **Collect** — `mlflow.log_feedback` from your UI captures human judgements (7.2).
# MAGIC - **Calibrate** — judge instructions tuned against human labels until agreement is high enough (7.3).
# MAGIC - **Enforce** — calibrated judges run as a Workflow gate that blocks deploys on regression (7.4).
# MAGIC - Together: human feedback → calibrated judges → automated gate → audited deploy.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Module 8** — closing the agent-improvement loop with prompt optimisation and regression catches.
# MAGIC