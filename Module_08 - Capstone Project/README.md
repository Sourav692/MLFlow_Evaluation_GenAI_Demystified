# Module 08 — Capstone: Full Enterprise Evaluation Pipeline

Six notebooks, ~2.5 hours end-to-end. The capstone composes every layer from Modules 01–07 into one closed-loop pipeline — **app → gateway → traces → judges → feedback → gate → next deploy** — all governed by Unity Catalog and auditable end-to-end.

## Labs

| # | Notebook | Layer | What you build |
| --- | --- | --- | --- |
| 1 | [8.1_Capstone_App_RAG_With_Gateway](8.1_Capstone_App_RAG_With_Gateway.ipynb) | App | RAG agent on Vector Search → registered to UC → deployed to Model Serving with AI Gateway (PII + Safety) and inference table |
| 2 | [8.2_Capstone_Hybrid_Eval_Dataset](8.2_Capstone_Hybrid_Eval_Dataset.ipynb) | Dataset | `generate_eval_df` + production traces from inference table + hand-curated edge cases → `tutorial_capstone_eval_v1` |
| 3 | [8.3_Capstone_Full_Scorer_Suite](8.3_Capstone_Full_Scorer_Suite.ipynb) | Scorers | `Correctness` + `RetrievalGroundedness` + `Safety` + `Guidelines` + `@scorer` latency + `make_judge` UC accuracy |
| 4 | [8.4_Capstone_CI_Quality_Gate](8.4_Capstone_CI_Quality_Gate.ipynb) | Gate | `to_predict_fn("endpoints:/...")` + versioned `THRESHOLDS` + `dbutils.notebook.exit("QUALITY_GATE_FAILED")` + Workflow YAML |
| 5 | [8.5_Capstone_Production_Monitoring](8.5_Capstone_Production_Monitoring.ipynb) | Monitoring | `.register().start(sample_rate=...)` + 50 simulated prod requests + verify `Assessment` rows on traces |
| 6 | [8.6_Capstone_Closed_Feedback_Loop](8.6_Capstone_Closed_Feedback_Loop.ipynb) | Feedback | `mlflow.log_feedback` → judge-vs-human agreement → v2 judge → re-gate → single-query audit |

## Outcome

> A complete enterprise-grade MLflow GenAI evaluation system on Databricks — covering dataset management, all scorer types, pre-deployment quality gates, AI Gateway guardrails, inference table monitoring, and a closed human feedback loop. **Every layer is governed by Unity Catalog and auditable end-to-end.**

The closed loop:

```text
deploy → gateway → traces → registered judges → human feedback → judge iteration → gate → next deploy
```

## Run order

1. **8.1 first** — it provisions the catalog, schema, Vector Search index, registered model, serving endpoint, and AI Gateway. The constants it persists (`CATALOG`, `SCHEMA`, `ENDPOINT_NAME`, `INDEX_FQN`, `INFERENCE_TABLE_FQN`) are read by every later notebook.
2. **8.2 → 8.6 in order.** Each is self-contained but depends on artefacts from earlier notebooks.
3. **(Optional)** Wire **8.4** as Task 1 of a Databricks Workflow whose Task 2 is the deploy step. The YAML snippet inside 8.4 shows the exact `depends_on` structure.

## Prerequisites

- Modules 01–07 complete, or comfort with the concepts they cover.
- A workspace with **Unity Catalog**, **Model Serving**, **Vector Search**, **Foundation Model API**, and **AI Gateway** all enabled.
- Permission to create UC objects, register models, deploy endpoints, and create Workflows.

## Companion material

The narrated walkthrough at [`../mlflow-genai-complete-tutorial.html`](../mlflow-genai-complete-tutorial.html) mirrors the capstone storyline.
