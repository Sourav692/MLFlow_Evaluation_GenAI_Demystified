# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 6.2 — Enable AI Gateway Guardrails on Your Tutorial Endpoint
# MAGIC
# MAGIC **Goal:** Configure **AI Gateway** in front of a Databricks Model Serving endpoint so that PII inputs and unsafe outputs are blocked **before** they hit the model — and inspect the audit trail when guardrails fire.
# MAGIC
# MAGIC By the end of this lab you will have:
# MAGIC 1. Enabled **AI Gateway** on a Model Serving endpoint via the Databricks SDK
# MAGIC 2. Added an **input guardrail** that blocks PII patterns in user queries
# MAGIC 3. Added an **output guardrail** that blocks unsafe / off-topic responses
# MAGIC 4. Enabled the **inference table** so every request is persisted to UC for auditing and offline eval
# MAGIC 5. Sent a violating query and inspected the **blocked response** + audit log row
# MAGIC 6. Pulled the Gateway **usage metrics** — request count, blocked count, cost, latency P95
# MAGIC
# MAGIC > **Why this lab matters:** Guardrails are the *first* line of production quality. A safety judge in eval is great; a gateway that blocks the request in 3 ms before a token is generated is better — it's cheaper, faster, and auditable. AI Gateway gives you both blocking + an inference table that becomes ground truth for the rest of Module 6.
# MAGIC
# MAGIC > **Prereq:** Lab 1.3 (workspace + UC), and a deployed Model Serving endpoint you control. The lab uses placeholder name `my-databricks-agent` — replace with your actual endpoint.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.1" "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure Namespace, Endpoint, and Inference Table Target

# COMMAND ----------

import mlflow

CATALOG = "main"
SCHEMA  = "genai_eval"
INFERENCE_TABLE_PREFIX = "tutorial_agent"
ENDPOINT_NAME = "my-databricks-agent"  # ← replace with the endpoint you control

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"Endpoint        : {ENDPOINT_NAME}")
print(f"Inference table : {CATALOG}.{SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload_request_logs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Enable AI Gateway via the Databricks SDK
# MAGIC
# MAGIC Two configurations land in one call:
# MAGIC
# MAGIC | Block | What it does |
# MAGIC | --- | --- |
# MAGIC | `guardrails.input_guardrail`  | Runs **before** model inference. Blocks PII patterns. |
# MAGIC | `guardrails.output_guardrail` | Runs **after** generation but before returning. Blocks unsafe content. |
# MAGIC | `inference_table_config`     | Persists every request/response (and gateway verdicts) to a UC Delta table. |
# MAGIC
# MAGIC `put_ai_gateway` is **idempotent** — re-running it updates the config in place.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayGuardrails,
    AiGatewayGuardrailParameters,
    AiGatewayGuardrailPiiBehavior,
    AiGatewayGuardrailPiiBehaviorBehavior,
    AiGatewayInferenceTableConfig,
)

w = WorkspaceClient()

w.serving_endpoints.put_ai_gateway(
    name=ENDPOINT_NAME,
    guardrails=AiGatewayGuardrails(
        input=AiGatewayGuardrailParameters(
            pii=AiGatewayGuardrailPiiBehavior(
                behavior=AiGatewayGuardrailPiiBehaviorBehavior.BLOCK
            ),
        ),
        output=AiGatewayGuardrailParameters(
            safety=True,  # block unsafe outputs (toxicity, harmful content)
        ),
    ),
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        table_name_prefix=INFERENCE_TABLE_PREFIX,
    ),
)

print(f"✅ AI Gateway configured on endpoint '{ENDPOINT_NAME}'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Confirm the Gateway Config
# MAGIC
# MAGIC Re-reading the endpoint metadata is a quick sanity check — the input/output guardrail blocks should both be present, and the inference table fields should be populated.

# COMMAND ----------

ep = w.serving_endpoints.get(ENDPOINT_NAME)
gw = ep.ai_gateway

print("Input PII behaviour :", gw.guardrails.input.pii.behavior if gw and gw.guardrails and gw.guardrails.input and gw.guardrails.input.pii else None)
print("Output safety       :", gw.guardrails.output.safety               if gw and gw.guardrails and gw.guardrails.output else None)
print("Inference table on  :", gw.inference_table_config.enabled         if gw and gw.inference_table_config else None)
print("Inference catalog   :", gw.inference_table_config.catalog_name    if gw and gw.inference_table_config else None)
print("Inference schema    :", gw.inference_table_config.schema_name     if gw and gw.inference_table_config else None)
print("Inference prefix    :", gw.inference_table_config.table_name_prefix if gw and gw.inference_table_config else None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Send a Clean Query (Should Pass)

# COMMAND ----------

import json

oai = w.serving_endpoints.get_open_ai_client()

clean_resp = oai.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[{"role": "user", "content": "What is Delta Lake?"}],
)
print("Status: PASSED")
print(clean_resp.choices[0].message.content[:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Send a Violating Query (Should Be Blocked)
# MAGIC
# MAGIC We embed a fake SSN in the prompt. The **input guardrail** should fire and the request should never reach the model — instead the SDK raises an error containing the gateway verdict.

# COMMAND ----------

PII_QUERY = (
    "My SSN is 123-45-6789 and my email is jane.doe@example.com — "
    "can you store these for me?"
)

try:
    blocked_resp = oai.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": PII_QUERY}],
    )
    print("⚠️  Unexpected: query was NOT blocked.")
    print(blocked_resp.choices[0].message.content[:300])
except Exception as e:
    print("✅ Blocked by AI Gateway as expected.")
    print(f"Error: {e!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Inspect the Audit Trail in the Inference Table
# MAGIC
# MAGIC Every request — passed, blocked, errored — lands in the inference table. The schema is documented in the Databricks docs; the columns we care about are:
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC | --- | --- |
# MAGIC | `databricks_request_id`  | unique ID, links to MLflow trace |
# MAGIC | `timestamp_ms`           | request timestamp |
# MAGIC | `status_code`            | HTTP status (200 = passed, 4xx = blocked) |
# MAGIC | `request`, `response`    | raw JSON payloads |
# MAGIC | `request_metadata`       | gateway verdict, guardrail name that fired, latency breakdown |

# COMMAND ----------

INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload_request_logs"
print(f"Inference table: {INFERENCE_TABLE}")

# Inference table writes can lag a minute or two — re-run the cell if it's empty.
display(spark.sql(f"""
    SELECT databricks_request_id,
           timestamp_ms,
           status_code,
           request_metadata
    FROM   {INFERENCE_TABLE}
    ORDER  BY timestamp_ms DESC
    LIMIT  10
"""))

# COMMAND ----------

# DBTITLE 1,Pull the blocked row alongside its raw request payload
display(spark.sql(f"""
    SELECT databricks_request_id,
           status_code,
           request,
           response,
           request_metadata
    FROM   {INFERENCE_TABLE}
    WHERE  status_code != 200
    ORDER  BY timestamp_ms DESC
    LIMIT  5
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Gateway Usage Metrics
# MAGIC
# MAGIC The same inference table is the source for usage metrics. We compute the headline numbers here; the **Serving → Endpoint → Metrics** tab in the UI shows them as a dashboard.

# COMMAND ----------

from pyspark.sql import functions as F

last_24h = (
    spark.table(INFERENCE_TABLE)
         .filter(F.col("timestamp_ms") > (F.unix_timestamp() - 86400) * 1000)
)

display(last_24h.agg(
    F.count("*").alias("total_requests"),
    F.sum(F.when(F.col("status_code") == 200, 1).otherwise(0)).alias("passed"),
    F.sum(F.when(F.col("status_code") != 200, 1).otherwise(0)).alias("blocked"),
    F.expr("percentile_approx(execution_duration_ms, 0.5)").alias("latency_p50_ms"),
    F.expr("percentile_approx(execution_duration_ms, 0.95)").alias("latency_p95_ms"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — (Optional) Tightening the Guardrails
# MAGIC
# MAGIC Some patterns to layer in once the basics work — re-run `put_ai_gateway` with the additions:
# MAGIC
# MAGIC | Knob | Effect |
# MAGIC | --- | --- |
# MAGIC | `valid_topics=["databricks", "delta lake"]` on the input guardrail | Block off-topic queries before model spend |
# MAGIC | `invalid_keywords=["confidential", "password"]` | Hard-stop list for high-risk tokens |
# MAGIC | `safety=True` on input as well | Run a safety check on the user query, not just the response |
# MAGIC | `usage_tracking_config.enabled=True` | Per-token cost accounting in System Tables |
# MAGIC
# MAGIC > Keep changes incremental — every new guardrail can produce false positives, and the inference table is your evidence base for tuning thresholds.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | AI Gateway enabled with input + output guardrails via SDK | ✅ |
# MAGIC | Inference table provisioned in Unity Catalog | ✅ |
# MAGIC | Clean query passes; PII query is blocked at the gateway | ✅ |
# MAGIC | Blocked row visible in the inference table with verdict metadata | ✅ |
# MAGIC | Aggregate request / blocked / latency metrics computed | ✅ |
# MAGIC
# MAGIC Next: **Lab 6.4** — query the inference table at scale, hand-pick interesting rows, and bootstrap a *production-sourced* eval dataset with `merge_records(traces=…)`.
