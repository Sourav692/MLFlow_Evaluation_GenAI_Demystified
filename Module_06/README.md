# Module 06 — Production: Gateway, Inference Tables, Monitoring

Pre-deployment eval tells you the model was good *yesterday*. Module 06 closes the gap to prod: front the endpoint with **AI Gateway** (PII + Safety guardrails), capture every request/response in an **inference table**, and let **registered scorers** grade live traffic asynchronously.

## Labs

| Notebook | What you do |
| --- | --- |
| [6.2_Lab_AI_Gateway_Guardrails](6.2_Lab_AI_Gateway_Guardrails.ipynb) | Configure AI Gateway on a Model Serving endpoint — PII detection on input, Safety on output, rate limits — and observe the gateway-level rejections |
| [6.4_Lab_Inference_Tables_Production_Dataset](6.4_Lab_Inference_Tables_Production_Dataset.ipynb) | Enable inference tables, query the captured Delta table, and convert recent rows into an MLflow eval dataset using `mlflow.search_traces` |
| [6.5_Lab_Production_Monitoring_Registered_Scorers](6.5_Lab_Production_Monitoring_Registered_Scorers.ipynb) | `.register().start(sample_rate=...)` your scorers so MLflow grades production traces in the background — verify `Assessment` rows land on traces |

## Outcome

By the end of Module 06:

- A deployed endpoint with **gateway guardrails** *and* **traffic capture** — the two halves of an audit trail.
- A production-derived eval dataset that updates every time you call `merge_records(traces=...)`.
- **Registered scorers running asynchronously on live traffic** at sampling rates you control — the foundation for ongoing quality dashboards.

## Prerequisites

- Module 05 (a working agent you can deploy, or an existing endpoint).
- Workspace permission to create serving endpoints and enable inference tables.

## Next

Module 07 — close the loop with human feedback and wire eval into a Workflow that gates deploys.
