# Module 07 — Quality Loop: Human Feedback and CI Gates

The two pieces that turn eval from a notebook ritual into an engineering discipline: **humans labelling traces** (so judges can be calibrated against ground truth) and **eval-as-CI** (so a regression blocks the deploy, not the next outage). Module 07 builds both.

## Labs

| Notebook | What you do |
| --- | --- |
| [7.2_Lab_Collect_User_Feedback](7.2_Lab_Collect_User_Feedback.ipynb) | Attach human labels to traces with `mlflow.log_feedback(...)`, compute judge-vs-human agreement, and identify the rows where the judge disagrees with the human |
| [7.4_Lab_Quality_Gate_Workflows](7.4_Lab_Quality_Gate_Workflows.ipynb) | Parameterise an eval notebook with widgets, threshold-check `results.metrics`, call `dbutils.notebook.exit("QUALITY_GATE_FAILED")` on regression, and wire it as Task 1 of a Workflow whose Task 2 is the deploy |

## Outcome

By the end of Module 07:

- A reproducible recipe for **judge calibration**: human labels in, agreement matrix out, judge-prompt iteration informed by disagreements.
- A real **pre-deployment quality gate** — a notebook task that hard-fails the Workflow on metric regression, with `depends_on` blocking the deploy step.
- Mental model: **eval is a code-review for the model** — humans set the bar, judges enforce it, the gate is the merge button.

## Prerequisites

- Module 04 (you understand judges and `make_judge`).
- Module 06 (registered scorers + inference table provide the traces to label).

## Next

Module 08 — the capstone. Compose every layer into one enterprise pipeline: app → gateway → traces → judges → feedback → gate → next deploy.
