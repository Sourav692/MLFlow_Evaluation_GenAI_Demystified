# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "1"
# ///
# MAGIC %md
# MAGIC # 🎯 Lab 4.4 — Custom LLM Judge with `make_judge()` on Databricks
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will learn:
# MAGIC 1. **Templated Instructions** — Use `{{ inputs }}` and `{{ outputs }}` placeholders inside a clear 1–5 rubric
# MAGIC 2. **Databricks-Hosted Judge Model** — Pin the judge LLM to `databricks:/databricks-claude-opus-4-6`
# MAGIC 3. **Per-Row Rationales** — Read rationales row-by-row to spot systemic judging bias
# MAGIC 4. **Calibration Loop** — Iterate the rubric (v1 → v2) to fix length bias, generic-correct bias, hallucination tolerance
# MAGIC 5. **Score Distribution Analysis** — Visualise how rubric changes shift the score distribution
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
# MAGIC ## Step 2 — Anatomy of a `make_judge()` Call
# MAGIC
# MAGIC `make_judge()` produces a scorer object you can pass into `evaluate()` like any built-in. Three things matter:
# MAGIC
# MAGIC 1. **`name`** — appears as the column prefix in eval results (`databricks_technical_accuracy/v1/value`).
# MAGIC 2. **`instructions`** — a templated string. `{{ inputs }}` and `{{ outputs }}` are substituted with each row's data. Include a step-by-step rubric and ask the model to reason before scoring.
# MAGIC 3. **`model`** — the LLM doing the grading. On Databricks, the canonical form is `databricks:/<endpoint-name>`.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 2 - ANATOMY OF A `MAKE_JUDGE()` CALL
# ============================================================================

from mlflow.genai.judges import make_judge

tech_accuracy = make_judge(
    name="databricks_technical_accuracy",
    instructions="""
Evaluate whether {{ outputs }} is technically accurate as an answer to the
Databricks / Delta Lake question in {{ inputs }}.

Score 1-5 using this rubric:
  5 = Precise, correct, includes important caveats (e.g. retention defaults, edition-specific limits)
  4 = Mostly correct, minor omissions
  3 = Partially correct, key concept present but incomplete or imprecise
  2 = Significant inaccuracy or misleading statement
  1 = Fundamentally wrong, unsafe to act on, or hallucinated APIs

Reason step-by-step before scoring. Cite the specific phrase that drove your score.
Return the integer score on the final line.
""",
    model="databricks:/databricks-claude-opus-4-6",
)

print("Custom judge ready:", tech_accuracy.name)


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Build a Test Agent and Run the Judge
# MAGIC
# MAGIC We use the simple Pattern 1 agent from Module 3 — the goal here is to score the agent, not engineer it.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 3 - BUILD A TEST AGENT AND RUN THE JUDGE
# ============================================================================

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

@mlflow.trace
def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": "You are a Databricks expert. Answer concisely."},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content


# COMMAND ----------

# ============================================================================
# ▶️ STEP
# ============================================================================

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)

results_v1 = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[tech_accuracy],
    # model_id="models:/my-agent/v1",
)

display(results_v1.tables["eval_results"])


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Inspect Per-Row Rationales
# MAGIC
# MAGIC The `rationale` column is where calibration happens. Read 5–10 rationales and ask:
# MAGIC - Is the score *aligned with how I would grade this row*?
# MAGIC - Is the rationale latching onto irrelevant features (e.g., answer length)?
# MAGIC - Are there cases where a 4 should have been a 3?
# MAGIC
# MAGIC Disagreement signals tell you what to add to the rubric in the next iteration.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 4 - INSPECT PER-ROW RATIONALES
# ============================================================================

import pandas as pd

df = results_v1.tables["eval_results"]

display(df[["request", "response", "databricks_technical_accuracy/value"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Iterate on the Rubric (Calibration Loop)
# MAGIC
# MAGIC A few common failure modes and how to fix them:
# MAGIC
# MAGIC | Symptom in rationales | Rubric change |
# MAGIC | --- | --- |
# MAGIC | "Long verbose answers get 5s even when wrong" | Add: *"Length and verbosity must not influence the score."* |
# MAGIC | "Generic correct-sounding answers grade too high" | Add: *"Score 3 or below if the answer applies generically to any data system."* |
# MAGIC | "Hallucinated APIs slip past with score 4" | Add an explicit fact: *"Score 1 if any API name does not exist in current Databricks docs."* |
# MAGIC | "Scores cluster at 3 — judge avoids extremes" | Force calibration: *"Use the full 1–5 range; if uncertain, prefer the lower score."* |
# MAGIC
# MAGIC Below is **v2** of the judge — same shape, more guardrails.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 5 - ITERATE ON THE RUBRIC (CALIBRATION LOOP)
# ============================================================================

tech_accuracy_v2 = make_judge(
    name="databricks_technical_accuracy",
    instructions="""
Evaluate whether {{ outputs }} is technically accurate as an answer to the
Databricks / Delta Lake question in {{ inputs }}.

Score 1-5 using this rubric:
  5 = Precise, correct, includes important caveats
  4 = Mostly correct, minor omissions
  3 = Partially correct, key concept present but incomplete
  2 = Significant inaccuracy or misleading statement
  1 = Fundamentally wrong, hallucinated APIs, or unsafe to act on

Calibration rules:
- Length and verbosity MUST NOT influence the score. A short correct answer scores higher than a long wrong one.
- If the answer would apply generically to any data system (no Databricks specifics), score 3 or lower.
- Score 1 if ANY API or feature name is wrong or does not exist in current Databricks documentation.
- Use the full 1-5 range. If uncertain, prefer the lower score.

Reason step-by-step. Cite the specific phrase that drove your score.
Return the integer score on the final line.
""",
    model="databricks:/databricks-claude-opus-4-6",
)

results_v2 = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[tech_accuracy_v2],
    # model_id="models:/my-agent/v1-judge-v2",
)

display(results_v2.tables["eval_results"][["request", "response", "databricks_technical_accuracy/value"]])


# COMMAND ----------

# ============================================================================
# ▶️ SCORE DISTRIBUTION: JUDGE V1 VS JUDGE V2 (SAME AGENT)
# ============================================================================

def score_dist(results, label):
    return (results.tables["eval_results"]
            .selectExpr(f"'{label}' AS judge_version", "`databricks_technical_accuracy/v1/value` AS score")
            .groupBy("judge_version", "score")
            .count())

display(score_dist(results_v1, "v1").union(score_dist(results_v2, "v2_calibrated")))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — (Stretch) Calibrate Against Human Labels
# MAGIC
# MAGIC The gold standard is **agreement with human labels**. If you have a small set of human-graded rows, compare judge output to humans:
# MAGIC - **Exact agreement** — % of rows where judge score == human score
# MAGIC - **Within-1 agreement** — % within ±1 of human (often more meaningful for 1–5 scales)
# MAGIC - **Cohen's kappa** — agreement adjusted for chance
# MAGIC
# MAGIC The block below is a sketch — drop in your own labelled rows when you have them. Module 6 covers human review in depth.
# MAGIC

# COMMAND ----------

# ============================================================================
# ▶️ STEP 6 - (STRETCH) CALIBRATE AGAINST HUMAN LABELS
# ============================================================================

# Pseudo-code:
#
# human_labels = spark.table("genai_eval_tutorial.module_01.human_grades_v1")  # request_id, human_score
# judge_scores = results_v2.tables["eval_results"].selectExpr(
#     "request_id", "CAST(`databricks_technical_accuracy/v1/value` AS INT) AS judge_score"
# )
# joined = judge_scores.join(human_labels, "request_id")
#
# from pyspark.sql.functions import abs as _abs, col, avg, when
# display(joined.agg(
#     avg(when(col("judge_score") == col("human_score"), 1.0).otherwise(0.0)).alias("exact_match"),
#     avg(when(_abs(col("judge_score") - col("human_score")) <= 1, 1.0).otherwise(0.0)).alias("within_1"),
# ))

print("Calibration block left as a stretch goal — Module 6 covers human review end-to-end.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | `make_judge()` called with `{{ inputs }}` / `{{ outputs }}` template | ✅ |
# MAGIC | Judge model pinned to `databricks:/databricks-claude-opus-4-6` | ✅ |
# MAGIC | Judge ran on tutorial dataset; per-row rationales inspected | ✅ |
# MAGIC | Rubric iterated (v1 → v2) based on observed failure modes | ✅ |
# MAGIC | Score-distribution shift between judge versions visualised | ✅ |
# MAGIC
# MAGIC Next: **Lab 4.5** — pass/fail compliance rules with the `Guidelines` judge, and compose all three scorer types into a single `evaluate()` call.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC
# MAGIC ### 1. Anatomy of `make_judge()`
# MAGIC - **`name`** — column prefix in eval results (`<name>/v1/value`, `<name>/v1/rationale`).
# MAGIC - **`instructions`** — templated string. `{{ inputs }}` and `{{ outputs }}` substitute per row.
# MAGIC - **`model`** — the grading LLM. On Databricks: `databricks:/<endpoint-name>`.
# MAGIC
# MAGIC ### 2. Calibration Loop
# MAGIC - 1. Run the judge, read 5–10 rationales.
# MAGIC - 2. Spot systemic biases (length bias, generic-correct bias, hallucination tolerance).
# MAGIC - 3. Add explicit calibration rules ("Length must not influence the score", "Score 1 if any API name doesn't exist").
# MAGIC - 4. Re-run, compare score distributions.
# MAGIC - 5. Validate against human labels in Module 6 before trusting in CI/prod.
# MAGIC
# MAGIC ### 3. When to Reach for `make_judge()`
# MAGIC - Domain-specific rubrics (technical accuracy in Databricks docs, clinical correctness, policy compliance).
# MAGIC - Graded scoring (1–5) where pass/fail is too coarse.
# MAGIC - Anywhere built-in judges feel too generic.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Lab 4.5** — pass/fail policy rules with `Guidelines`, then compose all three scorer types into one `evaluate()` call.
# MAGIC
