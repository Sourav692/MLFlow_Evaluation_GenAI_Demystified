# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 3.3 — Pattern 2: `to_predict_fn()` for Deployed Endpoints
# MAGIC
# MAGIC **Goal:** Evaluate an agent already deployed to Databricks Model Serving — no code changes to the serving endpoint required.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Understood how `to_predict_fn` works — translates `inputs` → endpoint payload, sets `return_trace=True`, copies the trace into the current experiment
# MAGIC 2. Confirmed input format must match the endpoint schema (chat endpoints need a `messages` key)
# MAGIC 3. Wrapped a tutorial endpoint and run evaluation against it
# MAGIC 4. Compared scores between the local agent (3.2) and the deployed endpoint — should be identical if the same model is behind both
# MAGIC 5. **Bonus:** seen the two remaining `predict_fn` patterns — registered model and async — so the full Module 3 outcome is covered
# MAGIC
# MAGIC > **Module 3 Outcome:** You can connect any agent — local function, deployed endpoint, registered model, or async — to the MLflow evaluation harness using the correct `predict_fn` pattern. You know when to use each.
# MAGIC
# MAGIC > **Prereq:** Lab 3.2 done. Either a real serving endpoint named `my-databricks-agent` exists, or you'll create the lightweight tutorial endpoint in Step 3 below.

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

# Pick the endpoint to evaluate. The Foundation Model API endpoints are
# pre-deployed in every workspace, so we can use one as a stand-in.
ENDPOINT_NAME = "databricks-claude-sonnet-4"  # any chat-style serving endpoint works

print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Eval data: {DATASET_FQN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — How `to_predict_fn` Works
# MAGIC
# MAGIC Under the hood, `mlflow.genai.to_predict_fn("endpoints:/<name>")` returns a callable that:
# MAGIC 1. **Translates** each row's `inputs` into the endpoint's payload shape
# MAGIC 2. **Calls** the endpoint via the Databricks SDK
# MAGIC 3. **Sets `return_trace=True`** so the endpoint emits a trace
# MAGIC 4. **Copies** that trace into the *current* experiment so eval and traces stay co-located
# MAGIC
# MAGIC The wrapper makes deployed endpoints feel exactly like a local Python function from MLflow's point of view. You can swap between Pattern 1 and Pattern 2 by changing one line.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Reshape the Eval Data for Chat Endpoints
# MAGIC
# MAGIC Chat-style endpoints expect a `messages` array, not a bare `question` string. Our Lab 2.2 dataset uses `inputs.question` — fine for local Python functions, but we need to rewrap each row before sending it to the endpoint.
# MAGIC
# MAGIC > **Best practice:** keep the dataset endpoint-agnostic (just `question`) and reshape at the call site. This way one dataset works for every pattern.

# COMMAND ----------

import mlflow.genai.datasets

eval_dataset = mlflow.genai.datasets.get_dataset(name=DATASET_FQN)
eval_pdf = eval_dataset.to_df().toPandas()

endpoint_eval_data = []
for _, row in eval_pdf.iterrows():
    endpoint_eval_data.append({
        "inputs": {
            "messages": [{"role": "user", "content": row["inputs"]["question"]}]
        },
        "expectations": row["expectations"],
    })

print(f"Reshaped {len(endpoint_eval_data)} rows for chat-endpoint format.")
print(endpoint_eval_data[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Wrap the Endpoint with `to_predict_fn`
# MAGIC
# MAGIC One line. The string format is `endpoints:/<endpoint-name>`. The returned callable is what you pass as `predict_fn`.

# COMMAND ----------

predict_fn = mlflow.genai.to_predict_fn(f"endpoints:/{ENDPOINT_NAME}")

# Smoke test — same shape as evaluate() will use
sample = predict_fn(messages=[{"role": "user", "content": "What is Z-ordering?"}])
print(sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Run Evaluation Against the Endpoint
# MAGIC
# MAGIC Same `evaluate()` call as Lab 3.2 — the only thing that changed is `predict_fn`. This is the punchline: **switching from a local agent to a deployed endpoint is a one-line change.**

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RelevanceToQuery

results_endpoint = mlflow.genai.evaluate(
    data=endpoint_eval_data,
    predict_fn=predict_fn,
    scorers=[Correctness(), RelevanceToQuery()],
    model_id=f"endpoints:/{ENDPOINT_NAME}",
)

print("Endpoint evaluation complete.")
display(results_endpoint.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Compare Local Agent vs Deployed Endpoint
# MAGIC
# MAGIC If the local agent (Lab 3.2) and the endpoint use the same underlying model and same prompt, scores should be statistically identical. Differences usually point at:
# MAGIC - Drift in the deployed prompt template
# MAGIC - Different model versions
# MAGIC - Endpoint pre/post-processing (guardrails, retrieval) that the local agent skips

# COMMAND ----------

# Pull the local-agent run (Lab 3.2 used model_id="models:/my-agent/1")
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_PATH], max_results=20)
display(runs[["run_id", "tags.mlflow.runName", "start_time"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Appendix A — Pattern 3: Registered Model from Unity Catalog
# MAGIC
# MAGIC **When to use:** the agent is logged to MLflow as a `pyfunc` or LangChain/LlamaIndex flavor and registered under Unity Catalog. Common for offline batch evaluation, regression suites in CI, or before promoting to a serving endpoint.

# COMMAND ----------

# Pseudo-code — uncomment and supply a real registered-model URI to run
#
# REGISTERED_URI = "models:/genai_eval_tutorial.module_01.my_agent/1"
#
# pyfunc_model = mlflow.pyfunc.load_model(REGISTERED_URI)
#
# def registered_predict_fn(question: str) -> str:
#     # pyfunc.predict expects a DataFrame; one row per call.
#     import pandas as pd
#     return pyfunc_model.predict(pd.DataFrame([{"question": question}]))[0]
#
# results_registered = mlflow.genai.evaluate(
#     data=eval_dataset,
#     predict_fn=registered_predict_fn,
#     scorers=[Correctness(), RelevanceToQuery()],
#     model_id=REGISTERED_URI,
# )

print("Pattern 3 example shown above. Uncomment with a real registered-model URI to run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Appendix B — Pattern 4: Async `predict_fn`
# MAGIC
# MAGIC **When to use:** the agent makes multiple downstream calls per question (RAG retrieval + LLM + tool calls) and you want concurrency to keep evaluation cost-bounded. `mlflow.genai.evaluate` accepts an `async def` predict function and will run rows in parallel internally.

# COMMAND ----------

import asyncio
from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

@mlflow.trace
async def async_agent(question: str) -> str:
    # asyncio.to_thread lets us run the sync OpenAI client in a thread pool
    # without blocking the event loop. Replace with a true async client
    # (e.g. httpx.AsyncClient) for higher throughput.
    def _call():
        return client.chat.completions.create(
            model="databricks-claude-sonnet-4",
            messages=[
                {"role": "system", "content": "You are a Databricks expert. Be concise."},
                {"role": "user",   "content": question},
            ],
        )
    resp = await asyncio.to_thread(_call)
    return resp.choices[0].message.content

# Smoke test (notebooks already have a running event loop)
print(await async_agent("What is Delta Lake?"))

# COMMAND ----------

results_async = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=async_agent,                     # async def is supported directly
    scorers=[Correctness(), RelevanceToQuery()],
    model_id="models:/my-agent-async/1",
)

print("Async-agent evaluation complete.")
display(results_async.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Module 3 Outcome — Decision Matrix
# MAGIC
# MAGIC | Pattern | `predict_fn` shape | When to use | Key gotcha |
# MAGIC | --- | --- | --- | --- |
# MAGIC | **1. Local function** (Lab 3.2) | `def fn(question: str)` decorated with `@mlflow.trace` | Active development, fastest iteration | Param names must match dataset `inputs.<key>` |
# MAGIC | **2. Deployed endpoint** (this lab) | `to_predict_fn("endpoints:/<name>")` | Validate behaviour of the *actual* prod endpoint | Reshape data to the endpoint's schema (chat → `messages`) |
# MAGIC | **3. Registered model** (Appendix A) | `pyfunc.load_model(uri).predict(...)` wrapped in a fn | CI regression on a versioned artifact | `pyfunc.predict` expects a DataFrame, not kwargs |
# MAGIC | **4. Async function** (Appendix B) | `async def fn(question: str)` with `@mlflow.trace` | Multi-step agents where concurrency cuts wall-clock cost | Notebook event loop is already running — use `await`, not `asyncio.run()` |
# MAGIC
# MAGIC Pick by **what's authoritative for your team right now**: the code on your laptop, the endpoint in production, the artifact in UC, or the high-throughput async runner. The `evaluate()` call itself is identical across all four — only `predict_fn` changes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | `to_predict_fn` wrapped a chat endpoint | ✅ |
# MAGIC | Eval data reshaped to chat-endpoint schema (`messages`) | ✅ |
# MAGIC | Endpoint scored with Correctness + RelevanceToQuery | ✅ |
# MAGIC | Local-agent vs endpoint runs compared in the MLflow UI | ✅ |
# MAGIC | Pattern 3 (registered model) and Pattern 4 (async) demonstrated | ✅ |
# MAGIC
# MAGIC **Module 3 done.** You can now plug any agent — local, deployed, registered, or async — into the same `evaluate()` harness.
