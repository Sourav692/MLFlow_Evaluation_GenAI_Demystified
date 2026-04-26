# MLflow Evaluation for GenAI — Demystified

A hands-on, end-to-end tutorial for building **production-grade evaluation systems for GenAI applications** on Databricks using MLflow 3. Eight modules walk you from a clean workspace to a closed-loop enterprise eval pipeline — every layer governed by Unity Catalog and auditable end-to-end.

---

## What you'll build

By the end of the capstone (Module 8), you will have built and operated:

- A **traced RAG agent** on Databricks Vector Search, deployed to Model Serving with **AI Gateway** guardrails and an **inference table**.
- A **UC-backed eval dataset** assembled from synthetic, production, and hand-curated rows.
- A complete **scorer suite** combining built-in judges, `Guidelines`, custom `@scorer` code, and `make_judge` LLM rubrics.
- A **pre-deployment quality gate** wired into a Databricks Workflow that blocks deploys on regression.
- **Production monitoring** via registered scorers running asynchronously at configured sample rates.
- A **closed human-feedback loop** where SME labels drive judge iteration and re-gating.

---

## Module map

| Module | Theme | Key labs |
| --- | --- | --- |
| **01** — Setup | Workspace, Unity Catalog, FM API | `1.3_Lab_Databricks_Workspace_Setup` |
| **02** — Datasets | UC-backed, production traces, synthetic | `2.2`, `2.3`, `2.4` |
| **03** — Eval Patterns | Local agent vs deployed endpoint | `3.2`, `3.3` |
| **04** — Scorers | Built-in judges, code scorers, custom LLM judge, Guidelines | `4.2`, `4.3`, `4.4`, `4.5` |
| **05** — RAG & Agents | Vector Search, LangGraph, tool-call efficiency | `5.2`, `5.2b`, `5.4`, `5.4b` |
| **06** — Production | AI Gateway, inference tables, registered monitoring | `6.2`, `6.4`, `6.5` |
| **07** — Quality Loop | User feedback, judge calibration, Workflow gates | `7.2`, `7.4` |
| **08** — Capstone | Full enterprise pipeline from scratch | `8.1`–`8.6` |

Each module folder holds Jupyter notebooks (`.ipynb`) that run on Databricks. Source `.py` versions of selected labs live in [`src/`](src/) for diffability and Asset Bundle deployment.

---

## Module 8 — Capstone

The capstone composes everything into one pipeline. Six notebooks, ~2.5 hours end-to-end, each self-contained:

| # | Notebook | Layer | What you build |
| --- | --- | --- | --- |
| 1 | [8.1_Capstone_App_RAG_With_Gateway](Module_08%20-%20Capstone%20Project/8.1_Capstone_App_RAG_With_Gateway.ipynb) | App | RAG agent on Vector Search → registered to UC → deployed to Model Serving with AI Gateway (PII + Safety) and inference table |
| 2 | [8.2_Capstone_Hybrid_Eval_Dataset](Module_08%20-%20Capstone%20Project/8.2_Capstone_Hybrid_Eval_Dataset.ipynb) | Dataset | `generate_eval_df` + production traces from inference table + hand-curated edge cases → `tutorial_capstone_eval_v1` |
| 3 | [8.3_Capstone_Full_Scorer_Suite](Module_08%20-%20Capstone%20Project/8.3_Capstone_Full_Scorer_Suite.ipynb) | Scorers | `Correctness` + `RetrievalGroundedness` + `Safety` + `Guidelines` + `@scorer` latency + `make_judge` UC accuracy |
| 4 | [8.4_Capstone_CI_Quality_Gate](Module_08%20-%20Capstone%20Project/8.4_Capstone_CI_Quality_Gate.ipynb) | Gate | `to_predict_fn("endpoints:/...")` + versioned `THRESHOLDS` + `dbutils.notebook.exit("QUALITY_GATE_FAILED")` + Workflow YAML |
| 5 | [8.5_Capstone_Production_Monitoring](Module_08%20-%20Capstone%20Project/8.5_Capstone_Production_Monitoring.ipynb) | Monitoring | `.register().start(sample_rate=...)` + 50 simulated prod requests + verify `Assessment` on traces |
| 6 | [8.6_Capstone_Closed_Feedback_Loop](Module_08%20-%20Capstone%20Project/8.6_Capstone_Closed_Feedback_Loop.ipynb) | Feedback | `mlflow.log_feedback` → judge-vs-human agreement → v2 judge → re-gate → single-query audit |

### Capstone outcome

> A complete enterprise-grade MLflow GenAI evaluation system on Databricks — covering dataset management, all scorer types, pre-deployment quality gates, AI Gateway guardrails, inference table monitoring, and a closed human feedback loop. **Every layer is governed by Unity Catalog and auditable end-to-end.**

The closed loop:

```text
deploy → gateway → traces → registered judges → human feedback → judge iteration → gate → next deploy
```

---

## Prerequisites

- **Databricks workspace** with Unity Catalog, Model Serving, Vector Search, and Foundation Model API enabled.
- **Python 3.10+** locally (only required if you want to author notebooks outside Databricks).
- Permission to create UC catalogs/schemas, register models, run Workflows, and create serving endpoints.
- Familiarity with MLflow 3 GenAI APIs (`mlflow.genai.evaluate`, `mlflow.genai.scorers`, `mlflow.genai.judges`, `mlflow.genai.datasets`).

---

## Repo layout

```text
.
├── Module_01/                       # Workspace + UC setup
├── Module_02/                       # Eval datasets (UC-backed, prod traces, synthetic)
├── Module_03/                       # Local vs deployed eval patterns
├── Module_04/                       # All four scorer types
├── Module_05/                       # RAG + agent evaluation
├── Module_06/                       # AI Gateway + inference tables + monitoring
├── Module_07/                       # Feedback loop + Workflow gates
├── Module_08 - Capstone Project/    # End-to-end capstone (6 notebooks)
├── src/                             # Databricks-source-format .py mirrors
├── databricks.yml                   # Asset Bundle definition
├── mlflow-genai-complete-tutorial.html  # Companion HTML tutorial
└── README.md
```

---

## Running the capstone

1. Open the workspace and clone this repo (Repos → Add Repo, or `databricks repos`).
2. Run **8.1** first — it provisions the catalog, schema, Vector Search index, registered model, serving endpoint, and AI Gateway. The constants it produces are read by every later notebook.
3. Run **8.2 → 8.6** in order. Each is self-contained but depends on artefacts from earlier notebooks.
4. (Optional) Wire **8.4** as Task 1 of a Databricks Workflow whose Task 2 is the deploy step. The YAML snippet inside 8.4 shows the exact structure with `depends_on`.

Total runtime: ~2.5 hours including endpoint cold starts.

---

## Conventions

- **Title cell** lists Learning Objectives + Prerequisites.
- **Banner comments** (`# === SECTION ===`) precede every code cell.
- **Step headings** (`## Step N — …`) split major phases.
- **`Lab Complete` checklist** + **Summary** at the end of every notebook.
- All FQNs follow `{catalog}.{schema}.{object}` Unity Catalog conventions.

---

## Companion material

- [`mlflow-genai-complete-tutorial.html`](mlflow-genai-complete-tutorial.html) — narrated walkthrough mirroring the eight modules.
