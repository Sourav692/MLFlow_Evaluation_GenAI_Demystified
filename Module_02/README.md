# Module 02 — Eval Datasets

Eval is only as good as the dataset behind it. Module 02 walks through the **three sources** of eval rows and the canonical **UC-backed dataset** API in MLflow GenAI.

## Labs

| Notebook | What you do |
| --- | --- |
| [2.2_Lab_UC_Backed_Dataset_From_Scratch](2.2_Lab_UC_Backed_Dataset_From_Scratch.ipynb) | Create `tutorial_eval_v1` via `mlflow.genai.datasets.create_dataset(...)`, hand-author 10 rows with `inputs` + `expectations`, and `merge_records(...)` them in |
| [2.3_Lab_Dataset_From_Production_Traces](2.3_Lab_Dataset_From_Production_Traces.ipynb) | Pull recent traces with `mlflow.search_traces(...)` and `merge_records(traces=...)` to bootstrap an eval dataset from real traffic |
| [2.4_Lab_Synthetic_Dataset_Generation](2.4_Lab_Synthetic_Dataset_Generation.ipynb) | Use `mlflow.genai.datasets.generate_eval_df(...)` to produce grounded `(question, expected_facts)` rows from a doc corpus |

## Outcome

By the end of Module 02:

- A registered, versioned, governable **UC-backed eval dataset** ready for any downstream `mlflow.genai.evaluate(...)` call.
- Three independent ways to seed rows: hand-curated, production-derived, synthetic.
- Mental model: **`inputs` + `expectations` + `tags` is the canonical schema** — every later module reads it.

## Prerequisites

- Module 01 complete (catalog/schema + experiment exist).

## Next

Module 03 — point an eval run at either an in-notebook agent or a deployed endpoint.
