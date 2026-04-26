# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4.5 — Guidelines Judge for Compliance & Style Rules
# MAGIC
# MAGIC **Goal:** Pass/fail compliance rules — the fastest way to enforce tone, format, and policy requirements. Then compose all three scorer types (built-in, code-based, custom LLM) in a single `evaluate()` call to deliver the Module 4 outcome.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Written 3 compliance rules for the tutorial agent and run them via `Guidelines(guidelines=[...])`
# MAGIC 2. Used `ExpectationsGuidelines()` for *per-row* rules supplied by domain experts
# MAGIC 3. Combined `Guidelines` + `Correctness` + a `@scorer` code check + a custom `make_judge()` LLM judge in one call
# MAGIC 4. Mapped every scorer type to the problem class it solves best (the **Module 4 outcome decision matrix**)
# MAGIC
# MAGIC > **Module 4 Outcome:** *You can compose any combination of built-in judges, code-based scorers (with `@scorer`), and custom LLM judges in a single `evaluate()` call. You know which type to use for which problem, and how to calibrate custom judges.*
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 + 2.2 + 4.2 + 4.3 + 4.4.

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
# MAGIC ## Step 2 — Why `Guidelines` Sits Between Built-ins and Custom Judges
# MAGIC
# MAGIC `Guidelines` is the lowest-friction LLM judge in MLflow:
# MAGIC - You write rules in **plain English** — no rubric, no template variables.
# MAGIC - It returns **pass/fail per rule** with rationale.
# MAGIC - Use it for tone, format, and policy compliance.
# MAGIC
# MAGIC When `Guidelines` isn't enough — when you need a graded rubric, weighted criteria, or domain reasoning — graduate to `make_judge()` (Lab 4.4).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Write Three Compliance Rules
# MAGIC
# MAGIC Realistic rules for the tutorial Databricks Q&A agent. The first is a **safety** rule, the second a **tone** rule, the third a **format** rule.

# COMMAND ----------

from mlflow.genai.scorers import Guidelines

compliance_rules = Guidelines(
    name="compliance",
    guidelines=[
        "The response must NOT recommend deleting production tables, dropping catalogs, or any irreversible destructive action.",
        "The response must be in English.",
        "The response must reference at least one specific Databricks feature, product, or documentation concept (e.g. Unity Catalog, Delta Lake, Photon, Lakehouse Federation).",
    ],
)

print("Compliance Guidelines scorer ready:", compliance_rules.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Build the Tutorial Agent

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

@mlflow.trace
def my_agent(question: str) -> str:
    resp = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[
            {"role": "system", "content": (
                "You are a Databricks expert. Answer concisely. "
                "Reference specific Databricks features (Delta Lake, Unity Catalog, etc.) where relevant. "
                "Never recommend destructive actions like dropping tables or catalogs."
            )},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Run `Guidelines` on the Eval Dataset

# COMMAND ----------

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)

results_guidelines = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[compliance_rules],
    model_id="models:/my-agent/v1-compliance",
)

display(results_guidelines.tables["eval_results"].select(
    "inputs",
    "outputs",
    "compliance/v1/value",
    "compliance/v1/rationale",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — `ExpectationsGuidelines` for Per-Row Rules
# MAGIC
# MAGIC Some rules apply to a *specific* row, not every row. Example: a question about retention should require mentioning "7 days"; a question about Photon should require mentioning "vectorized engine". Domain experts encode these as `expectations.guidelines` on the dataset row, and the `ExpectationsGuidelines` scorer reads them per-row.

# COMMAND ----------

from mlflow.genai.scorers import ExpectationsGuidelines

# Build a per-row dataset where each row carries its own guidelines.
per_row_data = [
    {
        "inputs": {"question": "Explain the VACUUM command in Delta Lake."},
        "expectations": {
            "guidelines": [
                "The response must mention the default retention period of 7 days.",
                "The response must warn that lowering retention can break time travel.",
            ]
        },
    },
    {
        "inputs": {"question": "How is Photon different from standard Spark execution?"},
        "expectations": {
            "guidelines": [
                "The response must mention that Photon is a vectorized engine written in C++.",
                "The response must mention compatibility with existing Spark APIs.",
            ]
        },
    },
    {
        "inputs": {"question": "What is Z-ordering in Delta Lake?"},
        "expectations": {
            "guidelines": [
                "The response must mention data skipping.",
                "The response must mention co-location of related data.",
            ]
        },
    },
]

results_per_row = mlflow.genai.evaluate(
    data=per_row_data,
    predict_fn=my_agent,
    scorers=[ExpectationsGuidelines()],
    model_id="models:/my-agent/v1-per-row-rules",
)

display(results_per_row.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Capstone: Compose ALL Three Scorer Types in One Call
# MAGIC
# MAGIC This is the Module 4 outcome made concrete. One `evaluate()` call, four scorer kinds, mixed freely:
# MAGIC
# MAGIC 1. **Built-in judge** → `Correctness` (semantic ground-truth check via `expected_facts`)
# MAGIC 2. **Built-in guidelines** → `Guidelines(...)` (English rules)
# MAGIC 3. **Code-based** → `@scorer` for citation presence (deterministic regex)
# MAGIC 4. **Custom LLM judge** → `make_judge()` for technical accuracy on a 1–5 rubric

# COMMAND ----------

import re
from mlflow.genai.scorers import Correctness, scorer
from mlflow.genai.judges import make_judge
from mlflow.entities import Feedback, AssessmentSource

CITATION_RE = re.compile(r"\b(Delta Lake|Unity Catalog|Photon|Lakehouse|Auto Loader|Delta Live Tables|MLflow)\b", re.IGNORECASE)

@scorer
def mentions_databricks_feature(outputs: str) -> Feedback:
    matches = set(m.lower() for m in CITATION_RE.findall(outputs))
    return Feedback(
        value="yes" if matches else "no",
        rationale=(f"Mentioned: {sorted(matches)}" if matches else "No Databricks features named"),
        source=AssessmentSource(source_type="CODE", source_id="feature_mention_v1"),
    )

tech_accuracy = make_judge(
    name="databricks_technical_accuracy",
    instructions="""
Evaluate whether {{ outputs }} is technically accurate as an answer to {{ inputs }}.
Score 1-5: 5 = precise with caveats; 3 = partially correct; 1 = wrong/hallucinated.
Length must NOT influence the score. Reason step-by-step before scoring.
Return the integer score on the final line.
""",
    model="databricks:/databricks-claude-sonnet-4",
)

results_capstone = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_agent,
    scorers=[
        Correctness(),                  # built-in LLM judge
        compliance_rules,               # built-in Guidelines judge (English rules)
        mentions_databricks_feature,    # @scorer code-based check
        tech_accuracy,                  # custom make_judge() LLM judge
    ],
    model_id="models:/my-agent/v1-capstone",
)

display(results_capstone.tables["eval_results"])

# COMMAND ----------

# DBTITLE 1,Capstone — aggregate per scorer type
display(results_capstone.tables["eval_results"].selectExpr(
    "AVG(CASE WHEN `correctness/v1/value`                   = 'yes' THEN 1.0 ELSE 0.0 END) AS builtin_correctness",
    "AVG(CASE WHEN `compliance/v1/value`                    = 'yes' THEN 1.0 ELSE 0.0 END) AS builtin_guidelines",
    "AVG(CASE WHEN `mentions_databricks_feature/v1/value`   = 'yes' THEN 1.0 ELSE 0.0 END) AS code_scorer",
    "AVG(CAST(`databricks_technical_accuracy/v1/value` AS DOUBLE))                              AS custom_judge_avg_score",
    "COUNT(*) AS rows",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Module 4 Outcome — Decision Matrix
# MAGIC
# MAGIC > **Outcome:** *You can compose any combination of built-in judges, code-based scorers (with `@scorer`), and custom LLM judges in a single `evaluate()` call. You know which type to use for which problem, and how to calibrate custom judges.*
# MAGIC
# MAGIC ### Which Scorer Type for Which Problem?
# MAGIC
# MAGIC | Problem class | Use | Why | Lab |
# MAGIC | --- | --- | --- | --- |
# MAGIC | Semantic correctness vs `expected_facts` | `Correctness()` (built-in) | Calibrated, free, generic | 4.2 |
# MAGIC | "Does the answer address the question?" | `RelevanceToQuery()` (built-in) | Generic, drop-in | 4.2 |
# MAGIC | "Are claims grounded in retrieved docs?" | `RetrievalGroundedness()` (built-in) | Reads `RETRIEVER` spans automatically | 4.2 |
# MAGIC | Harmful / biased / PII content | `Safety()` (built-in) | Pre-trained on safety taxonomies | 4.2 |
# MAGIC | English rules: tone, format, policy | `Guidelines(guidelines=[...])` | Pass/fail, no rubric needed | 4.5 |
# MAGIC | Per-row rules from domain experts | `ExpectationsGuidelines()` | Reads `expectations.guidelines` from dataset | 4.5 |
# MAGIC | Deterministic checks (JSON, regex, latency) | `@scorer` (code-based) | Cheap, exact, no LLM call | 4.3 |
# MAGIC | Trace-level checks (latency, span counts) | `@scorer` reading `trace.search_spans(...)` | Score lives in spans, not output text | 4.3 |
# MAGIC | Domain-specific graded rubric (1–5) | `make_judge()` (custom LLM) | Calibrated to your domain, full template control | 4.4 |
# MAGIC
# MAGIC ### Calibration Loop (for `make_judge`)
# MAGIC
# MAGIC 1. Run the judge, read 5–10 rationales.
# MAGIC 2. Spot systemic biases (length bias, generic-correct bias, hallucination tolerance).
# MAGIC 3. Add explicit calibration rules to the rubric ("Length must not influence the score", "Score 1 if any API name does not exist").
# MAGIC 4. Re-run, compare score distributions across judge versions.
# MAGIC 5. Validate against human labels (Module 6) before trusting the judge in CI/prod.
# MAGIC
# MAGIC ### Composition Rule
# MAGIC
# MAGIC > **You don't pick one scorer type — you compose them.** Code scorers catch hard violations cheaply, built-in judges grade generic quality, custom `make_judge()` judges grade domain quality, and `Guidelines` enforces policy. They share one `scorers=[...]` list, one `evaluate()` call, one results table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete — Module 4 Outcome Coverage
# MAGIC
# MAGIC | Outcome Component | Where Covered | Status |
# MAGIC | --- | --- | --- |
# MAGIC | Built-in judges in one `evaluate()` call | Lab 4.2 + Step 7 capstone here | ✅ |
# MAGIC | Code-based `@scorer` decorators | Lab 4.3 + Step 7 capstone here | ✅ |
# MAGIC | Custom LLM judge with `make_judge()` | Lab 4.4 + Step 7 capstone here | ✅ |
# MAGIC | Compose all three in one `evaluate()` call | **Step 7 capstone** above | ✅ |
# MAGIC | Know which type to use for which problem | **Decision matrix** above | ✅ |
# MAGIC | Know how to calibrate custom judges | Lab 4.4 + calibration loop above | ✅ |
# MAGIC
# MAGIC **Module 4 done.** You have the full scorer toolbox. Next: **Module 5 — Custom LLM Judges in Depth** (advanced rubric design, multi-turn evaluation, judge-of-judges patterns).
