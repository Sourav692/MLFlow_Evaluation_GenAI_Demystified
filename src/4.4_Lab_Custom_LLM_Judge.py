# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4.4 — Custom LLM Judge with `make_judge()` on Databricks
# MAGIC
# MAGIC **Goal:** Build a domain-specific "Databricks Technical Accuracy" judge using a Databricks-hosted model as the judge LLM, then iterate on the rubric.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Written judge instructions with `{{ inputs }}` and `{{ outputs }}` template variables and a clear 1–5 rubric
# MAGIC 2. Set the judge model to a Databricks endpoint: `model="databricks:/databricks-claude-sonnet-4"`
# MAGIC 3. Run the judge on the tutorial dataset and inspected per-row rationales
# MAGIC 4. Iterated on the rubric based on results — the **calibration loop**
# MAGIC 5. (Stretch) Calibrated the judge against ground-truth labels and computed agreement
# MAGIC
# MAGIC > **When to reach for a custom LLM judge:** any scoring rubric that's domain-specific (technical accuracy in Databricks docs, policy compliance for an insurer, clinical correctness for a healthcare bot). Built-in judges are intentionally generic.
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 + 2.2 + 4.2.

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
# MAGIC ## Step 2 — Anatomy of a `make_judge()` Call
# MAGIC
# MAGIC `make_judge()` produces a scorer object you can pass into `evaluate()` like any built-in. Three things matter:
# MAGIC
# MAGIC 1. **`name`** — appears as the column prefix in eval results (`databricks_technical_accuracy/v1/value`).
# MAGIC 2. **`instructions`** — a templated string. `{{ inputs }}` and `{{ outputs }}` are substituted with each row's data. Include a step-by-step rubric and ask the model to reason before scoring.
# MAGIC 3. **`model`** — the LLM doing the grading. On Databricks, the canonical form is `databricks:/<endpoint-name>`.

# COMMAND ----------

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
    model="databricks:/databricks-claude-sonnet-4",
)

print("Custom judge ready:", tech_accuracy.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build a Test Agent and Run the Judge
# MAGIC
# MAGIC We use the simple Pattern 1 agent from Module 3 — the goal here is to score the agent, not engineer it.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

@mlflow.trace
def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[
            {"role": "system", "content": "You are a Databricks expert. Answer concisely."},
            {"role": "user",   "content": question},
        ],
    )
    return resp.choices[0].message.content

# COMMAND ----------

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)

results_v1 = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[tech_accuracy],
    model_id="models:/my-agent/v1",
)

display(results_v1.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Inspect Per-Row Rationales
# MAGIC
# MAGIC The `rationale` column is where calibration happens. Read 5–10 rationales and ask:
# MAGIC - Is the score *aligned with how I would grade this row*?
# MAGIC - Is the rationale latching onto irrelevant features (e.g., answer length)?
# MAGIC - Are there cases where a 4 should have been a 3?
# MAGIC
# MAGIC Disagreement signals tell you what to add to the rubric in the next iteration.

# COMMAND ----------

display(results_v1.tables["eval_results"].select(
    "inputs",
    "outputs",
    "databricks_technical_accuracy/v1/value",
    "databricks_technical_accuracy/v1/rationale",
))

# COMMAND ----------

# MAGIC %md
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

# COMMAND ----------

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
    model="databricks:/databricks-claude-sonnet-4",
)

results_v2 = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[tech_accuracy_v2],
    model_id="models:/my-agent/v1-judge-v2",
)

display(results_v2.tables["eval_results"].select(
    "inputs",
    "databricks_technical_accuracy/v1/value",
    "databricks_technical_accuracy/v1/rationale",
))

# COMMAND ----------

# DBTITLE 1,Score distribution: judge v1 vs judge v2 (same agent)
def score_dist(results, label):
    return (results.tables["eval_results"]
            .selectExpr(f"'{label}' AS judge_version", "`databricks_technical_accuracy/v1/value` AS score")
            .groupBy("judge_version", "score")
            .count())

display(score_dist(results_v1, "v1").union(score_dist(results_v2, "v2_calibrated")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — (Stretch) Calibrate Against Human Labels
# MAGIC
# MAGIC The gold standard is **agreement with human labels**. If you have a small set of human-graded rows, compare judge output to humans:
# MAGIC - **Exact agreement** — % of rows where judge score == human score
# MAGIC - **Within-1 agreement** — % within ±1 of human (often more meaningful for 1–5 scales)
# MAGIC - **Cohen's kappa** — agreement adjusted for chance
# MAGIC
# MAGIC The block below is a sketch — drop in your own labelled rows when you have them. Module 6 covers human review in depth.

# COMMAND ----------

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
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | `make_judge()` called with `{{ inputs }}` / `{{ outputs }}` template | ✅ |
# MAGIC | Judge model pinned to `databricks:/databricks-claude-sonnet-4` | ✅ |
# MAGIC | Judge ran on tutorial dataset; per-row rationales inspected | ✅ |
# MAGIC | Rubric iterated (v1 → v2) based on observed failure modes | ✅ |
# MAGIC | Score-distribution shift between judge versions visualised | ✅ |
# MAGIC
# MAGIC Next: **Lab 4.5** — pass/fail compliance rules with the `Guidelines` judge, and compose all three scorer types into a single `evaluate()` call.
