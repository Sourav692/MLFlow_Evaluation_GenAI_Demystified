# Module 03 — Eval Patterns: Local Agent vs Deployed Endpoint

Two patterns dominate `mlflow.genai.evaluate(...)`: pointing at an **in-notebook agent function** during development, and pointing at a **deployed Model Serving endpoint** for staging/prod gates. Module 03 walks through both shapes side-by-side.

## Labs

| Notebook | What you do |
| --- | --- |
| [3.2_Lab_Pattern1_Local_Agent](3.2_Lab_Pattern1_Local_Agent.ipynb) | Define a `@mlflow.trace`-decorated agent function and pass it directly as `predict_fn=...` — the dev-loop pattern |
| [3.3_Lab_Pattern2_Deployed_Endpoint](3.3_Lab_Pattern2_Deployed_Endpoint.ipynb) | Use `mlflow.genai.scorers.to_predict_fn("endpoints:/...")` to evaluate a deployed serving endpoint — the gate/CI pattern |

## Outcome

By the end of Module 03:

- A clear contract: `predict_fn(inputs) -> outputs` is what `evaluate(...)` consumes — whether it's local or remote.
- Same dataset, same scorers, two execution targets — the pattern that makes Module 08's gate possible.
- Comfort reading per-row tables (`results.tables["eval_results"]`) and aggregate metrics (`results.metrics`).

## Prerequisites

- Module 02 (eval dataset registered).
- For 3.3: a deployed Model Serving endpoint you can hit (or use the one from a later module).

## Next

Module 04 — the four scorer types: built-in judges, code scorers, custom LLM judges, Guidelines.
