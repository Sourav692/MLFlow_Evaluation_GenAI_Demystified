# Module 05 — RAG and Agents

Eval gets interesting once the system under test is more than a single LLM call. Module 05 evaluates two real architectures: a **RAG pipeline** on Databricks Vector Search, and a **multi-step LangGraph agent** that plans, calls tools, and answers. Both flavours come in a "from scratch" notebook and a LangGraph variant.

## Labs

| Notebook | What you do |
| --- | --- |
| [5.2_Lab_RAG_Vector_Search](5.2_Lab_RAG_Vector_Search.ipynb) | Build a retrieve→augment→generate RAG agent on a Databricks Vector Search index, then evaluate with `Correctness` + `RetrievalGroundedness` + `RetrievalRelevance` |
| [5.2b_Lab_RAG_LangGraph](5.2b_Lab_RAG_LangGraph.ipynb) | Same RAG contract, expressed as a LangGraph state machine — shows how `@mlflow.trace` spans line up with graph nodes |
| [5.4_Lab_Tool_Call_Efficiency](5.4_Lab_Tool_Call_Efficiency.ipynb) | Evaluate a tool-calling agent with code scorers that count tool calls, detect redundant calls, and enforce a tool-call budget |
| [5.4b_Lab_Agent_LangGraph](5.4b_Lab_Agent_LangGraph.ipynb) | A LangGraph ReAct agent with bound tools — evaluate end-to-end accuracy *and* per-step trajectory quality |

## Outcome

By the end of Module 05:

- Two working agents (RAG + tool-using) traced into MLflow with span-level visibility.
- Confidence reading retriever-specific metrics: groundedness, retrieval-relevance, retrieved-chunk inspection.
- Patterns for grading **trajectory** (how the agent got there), not just **final answer**.

## Prerequisites

- Module 04 (you have the four scorer shapes).
- Vector Search endpoint + a populated index (the lab walks through provisioning if missing).

## Next

Module 06 — wrap one of these agents in AI Gateway, capture an inference table, and run scorers in production.
