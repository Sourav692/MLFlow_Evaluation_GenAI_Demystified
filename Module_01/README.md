# Module 01 — Workspace Setup

Bring a fresh Databricks workspace to a state where every later module can run: Unity Catalog enabled, Foundation Model API reachable, an MLflow experiment created, and your user identity wired up so traces and models land in the right place.

## Labs

| Notebook | What you do |
| --- | --- |
| [1.3_Lab_Databricks_Workspace_Setup](1.3_Lab_Databricks_Workspace_Setup.ipynb) | Confirm UC + FM API + Vector Search entitlements, create the tutorial catalog/schema, set up an MLflow experiment under your user, smoke-test the `mlflow.openai` client against a Databricks-served model |

## Outcome

By the end of Module 01:

- A working **catalog + schema** in Unity Catalog you own.
- An **MLflow experiment** at `/Users/<you>/genai-eval-tutorial`.
- A verified call to a Databricks Foundation Model endpoint via the OpenAI-compatible client.
- Tracing on by default — `mlflow.openai.autolog()` produces traces you can see in the experiment.

## Prerequisites

- Databricks workspace with **Unity Catalog**, **Foundation Model API**, and **Vector Search** enabled.
- Permission to create catalogs and schemas (or an existing UC namespace you can use).

## Next

Module 02 — build the canonical eval datasets that every later notebook reads.
