# Module 04 — The Four Scorer Types

`mlflow.genai.evaluate(...)` accepts a `scorers=[...]` list, and there are exactly **four shapes** that can go in it. Module 04 builds one of each, end-to-end, against the same eval dataset so you can see which questions each shape answers.

## Labs

| Notebook | What you do |
| --- | --- |
| [4.2_Lab_Builtin_Judges](4.2_Lab_Builtin_Judges.ipynb) | Run the prebuilt LLM judges — `Correctness`, `RelevanceToQuery`, `Safety`, `RetrievalGroundedness`, `RetrievalRelevance` — and read the `feedback.value` / `feedback.rationale` columns |
| [4.3_Lab_Code_Based_Scorers](4.3_Lab_Code_Based_Scorers.ipynb) | Write deterministic Python scorers with `@scorer` for things judges can't see — latency, token cost, schema validity, exact-match assertions |
| [4.4_Lab_Custom_LLM_Judge](4.4_Lab_Custom_LLM_Judge.ipynb) | Author a project-specific rubric with `make_judge(...)` — instructions, scale, examples — and reuse it across runs |
| [4.5_Lab_Guidelines_Judge](4.5_Lab_Guidelines_Judge.ipynb) | Express plain-English policy rules as `Guidelines(...)` — the lightest-weight way to enforce tone, format, refusal behaviour |

## Outcome

By the end of Module 04:

- A working scorer of every shape, all reading the same `inputs` + `expectations` schema.
- Mental model: **judges grade quality, code grades behaviour, custom judges grade domain rubrics, guidelines grade policy.**
- A composite `scorers=[...]` list you can drop into any later `evaluate(...)` call — Modules 06–08 all reuse it.

## Prerequisites

- Module 02 (eval dataset registered).
- Module 03 (you know how `predict_fn` is wired).

## Next

Module 05 — apply these scorers to a real RAG agent and a multi-step LangGraph agent.
