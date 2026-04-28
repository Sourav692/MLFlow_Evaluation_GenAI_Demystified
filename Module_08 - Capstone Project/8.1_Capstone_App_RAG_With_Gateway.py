# Databricks notebook source
# MAGIC %md
# MAGIC # 🏛️ Capstone 8.1 — RAG App on Vector Search + AI Gateway + Inference Table
# MAGIC
# MAGIC **Goal:** Build the **App layer** for the capstone — a traced RAG agent answering Unity Catalog governance questions, deployed to Model Serving with **AI Gateway guardrails** and an **inference table** turned on. Every later notebook in Module 8 reads from artefacts produced here.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you will:
# MAGIC 1. **Index UC Governance Docs** — Build a Vector Search index over a small corpus of Unity Catalog governance chunks
# MAGIC 2. **Build a Traced RAG Agent** — RETRIEVER span + LLM span via `mlflow.openai.autolog()` so every pillar of three-pillar eval is observable
# MAGIC 3. **Log and Register the Model** — `mlflow.pyfunc.log_model(...)` + `mlflow.register_model(...)` into Unity Catalog
# MAGIC 4. **Deploy to Model Serving** — Create / update a serving endpoint pointing at the registered model
# MAGIC 5. **Enable AI Gateway** — PII input guardrail + safety output guardrail + inference table to UC
# MAGIC 6. **Smoke-Test the Endpoint** — Send one clean query and one PII query; confirm guardrail blocks and inference table records both
# MAGIC 7. **Persist Capstone Constants** — Catalog/schema/endpoint/index identifiers that Notebooks 2-6 will reuse
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Lab 1.3 (workspace + Unity Catalog + FM API enabled)
# MAGIC - Lab 5.2 (Vector Search basics) and Lab 6.2 (AI Gateway basics)
# MAGIC - Vector Search and Model Serving entitled in this workspace
# MAGIC - Permission to create UC catalogs/schemas and register models
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 INSTALL PACKAGES
# ============================================================================

%pip install --quiet "mlflow[databricks]>=3.1" databricks-openai databricks-vectorsearch "databricks-sdk>=0.40"

dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Capstone Constants & Experiment
# MAGIC
# MAGIC Every Module 8 notebook references the same catalog, schema, endpoint, index, and inference table. Pin them once here and read them via `dbutils.jobs.taskValues` (or hardcoded constants) downstream so the whole pipeline is traceable end-to-end.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧭 STEP 1 - CAPSTONE CONSTANTS & EXPERIMENT
# ============================================================================

import mlflow

CATALOG = "genai_eval_tutorial"
SCHEMA  = "module_08_capstone"

DOCS_TABLE        = f"{CATALOG}.{SCHEMA}.uc_governance_chunks"
VS_ENDPOINT       = "genai_capstone_vs"
VS_INDEX_FQN      = f"{CATALOG}.{SCHEMA}.uc_governance_index"
REGISTERED_MODEL  = f"{CATALOG}.{SCHEMA}.uc_governance_rag"
SERVING_ENDPOINT  = "genai-capstone-rag"
INFERENCE_PREFIX  = "capstone_rag"
INFERENCE_TABLE   = f"{CATALOG}.{SCHEMA}.{INFERENCE_PREFIX}_payload_request_logs"

USER_EMAIL = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/genai-eval-tutorial"
mlflow.set_experiment(EXPERIMENT_PATH)

# Autolog every OpenAI-compatible call so retrieval embeddings + generation are traced.
mlflow.openai.autolog()

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"✅ Experiment       : {EXPERIMENT_PATH}")
print(f"📚 Docs table       : {DOCS_TABLE}")
print(f"🧠 VS index         : {VS_INDEX_FQN}")
print(f"🚀 Serving endpoint : {SERVING_ENDPOINT}")
print(f"🗃️  Inference table  : {INFERENCE_TABLE}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Author the UC Governance Doc Chunks
# MAGIC
# MAGIC The capstone is themed around **Unity Catalog governance**. We hand-author 10 short chunks covering the topics our eval set will ask about (catalog hierarchy, grants, lineage, masking, audit logs, etc.). In production these would come from a real ingestion pipeline.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📚 STEP 2 - AUTHOR UC GOVERNANCE DOC CHUNKS
# ============================================================================

from pyspark.sql import Row

chunks = [
    Row(doc_id="uc_hierarchy",       content="Unity Catalog organises data into a three-level namespace: catalog.schema.object. Catalogs are the top-level grouping for environments or business domains; schemas group related tables, views, volumes and models inside a catalog."),
    Row(doc_id="uc_grants",          content="Unity Catalog uses ANSI-style GRANT and REVOKE statements. Grants flow with the hierarchy: USE CATALOG and USE SCHEMA are required before SELECT on a table. The catalog owner can manage all grants beneath it."),
    Row(doc_id="uc_lineage",         content="Unity Catalog automatically captures column- and table-level lineage for jobs, notebooks, and SQL warehouses for up to one year. Lineage powers impact analysis when a column is renamed or a table is deprecated."),
    Row(doc_id="uc_dynamic_views",   content="Row- and column-level access control is enforced via dynamic views that wrap a base table with CASE expressions on is_member() or current_user(). Users only see the rows or columns their group entitlement allows."),
    Row(doc_id="uc_masking",         content="Column masks are SQL functions registered with ALTER TABLE ... ALTER COLUMN ... SET MASK. The function receives the raw value plus context functions (current_user, is_member) and returns the masked output. Masks compose with row filters."),
    Row(doc_id="uc_audit_logs",      content="Unity Catalog audit logs are delivered to system tables (system.access.audit). Every grant change, table read, and lineage event is recorded with actor, action, and resource — immutable for compliance review."),
    Row(doc_id="uc_external_tables", content="External tables are governed by Unity Catalog while their data files live in customer cloud storage. Storage credentials and external locations bind UC permissions to underlying object-store paths so policy is enforced at read time."),
    Row(doc_id="uc_volumes",         content="Volumes provide governed file-level access for unstructured data: PDFs, images, model weights. Read and write are governed by the same catalog.schema.volume grant model as tables."),
    Row(doc_id="uc_delta_sharing",   content="Delta Sharing publishes Unity Catalog tables, views and volumes to external recipients without copying data. Recipients receive short-lived credentials; the provider revokes access by removing the share grant."),
    Row(doc_id="uc_models",          content="Registered models live in Unity Catalog as catalog.schema.model. Aliases (champion, challenger) replace stages. EXECUTE permission is required to load a model; MANAGE to register a new version."),
]

(spark.createDataFrame(chunks)
       .write.mode("overwrite")
       .option("delta.enableChangeDataFeed", "true")
       .saveAsTable(DOCS_TABLE))

print(f"✅ Wrote {len(chunks)} chunks to {DOCS_TABLE}")
display(spark.table(DOCS_TABLE))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Vector Search Endpoint and Index
# MAGIC
# MAGIC We use a managed embedding model so we don't have to compute embeddings ourselves. Endpoint creation is idempotent but slow on first run — re-running is safe.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🔍 STEP 3 - VECTOR SEARCH ENDPOINT + INDEX
# ============================================================================

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"⏳ Creating endpoint {VS_ENDPOINT} (this may take a few minutes)…")
else:
    print(f"✅ Endpoint {VS_ENDPOINT} already exists.")

vsc.wait_for_endpoint(VS_ENDPOINT, timeout=1200)

try:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_FQN)
    print(f"✅ Index {VS_INDEX_FQN} already exists.")
except Exception:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX_FQN,
        source_table_name=DOCS_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="doc_id",
        embedding_source_column="content",
        embedding_model_endpoint_name="databricks-gte-large-en",
    )
    print(f"✅ Created index {VS_INDEX_FQN}.")

index.wait_until_ready(verbose=True, timeout=600)
print("🟢 Index is READY.")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Build the Traced RAG Agent
# MAGIC
# MAGIC Two spans: a `RETRIEVER` span (so `RetrievalGroundedness` and any custom recall scorer can read retrieved docs) and an LLM span (auto-traced by `mlflow.openai.autolog()` so `Correctness` can read the final output).
# MAGIC

# COMMAND ----------

# ============================================================================
# 🤖 STEP 4 - TRACED RAG AGENT
# ============================================================================

from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType

llm_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
TOP_K = 3
GEN_MODEL = "databricks-claude-opus-4-6"

SYSTEM_PROMPT = (
    "You are a Unity Catalog governance expert. Answer using ONLY the provided context. "
    "Cite the doc_id in square brackets after each claim. "
    "If the context does not answer the question, say so explicitly. Be concise."
)

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str) -> list[dict]:
    res = index.similarity_search(
        query_text=question,
        columns=["doc_id", "content"],
        num_results=TOP_K,
    )
    rows = res.get("result", {}).get("data_array", [])
    return [
        {"page_content": r[1], "metadata": {"doc_id": r[0], "score": r[2]}}
        for r in rows
    ]

@mlflow.trace
def rag_agent(question: str) -> str:
    docs = retrieve(question)
    context = "\n".join(f"[{d['metadata']['doc_id']}] {d['page_content']}" for d in docs)
    resp = llm_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content

print("🧪 Smoke test:")
print(rag_agent("How does Unity Catalog enforce row-level access control?"))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — Wrap as `mlflow.pyfunc` and Register to Unity Catalog
# MAGIC
# MAGIC Model Serving needs a registered MLflow model. We wrap the agent as a `PythonModel` and register it to UC at `genai_eval_tutorial.module_08_capstone.uc_governance_rag`.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📦 STEP 5 - LOG + REGISTER TO UNITY CATALOG
# ============================================================================

import mlflow.pyfunc

mlflow.set_registry_uri("databricks-uc")

class UCGovernanceRAG(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper around the traced RAG agent."""

    def load_context(self, context):
        # Re-create clients lazily inside the serving container.
        from databricks.sdk import WorkspaceClient
        from databricks.vector_search.client import VectorSearchClient
        self._llm   = WorkspaceClient().serving_endpoints.get_open_ai_client()
        self._index = VectorSearchClient(disable_notice=True).get_index(
            endpoint_name=context.model_config["vs_endpoint"],
            index_name=context.model_config["vs_index"],
        )
        self._gen_model = context.model_config["gen_model"]
        self._top_k     = context.model_config["top_k"]
        self._system    = context.model_config["system_prompt"]

    @mlflow.trace
    def predict(self, context, model_input):
        if hasattr(model_input, "to_dict"):
            questions = model_input["question"].tolist()
        else:
            questions = [model_input["question"]] if isinstance(model_input, dict) else list(model_input)
        answers = []
        for q in questions:
            res = self._index.similarity_search(
                query_text=q,
                columns=["doc_id", "content"],
                num_results=self._top_k,
            )
            rows = res.get("result", {}).get("data_array", [])
            ctx = "\n".join(f"[{r[0]}] {r[1]}" for r in rows)
            chat = self._llm.chat.completions.create(
                model=self._gen_model,
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
                ],
            )
            answers.append(chat.choices[0].message.content)
        return answers

with mlflow.start_run(run_name="capstone-rag-app") as run:
    mlflow.set_tags({"capstone": "module_08", "layer": "app"})
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=UCGovernanceRAG(),
        registered_model_name=REGISTERED_MODEL,
        model_config={
            "vs_endpoint":   VS_ENDPOINT,
            "vs_index":      VS_INDEX_FQN,
            "gen_model":     GEN_MODEL,
            "top_k":         TOP_K,
            "system_prompt": SYSTEM_PROMPT,
        },
        pip_requirements=[
            "mlflow[databricks]>=3.1",
            "databricks-sdk>=0.40",
            "databricks-vectorsearch",
        ],
        input_example={"question": ["What is a Unity Catalog volume?"]},
    )
    APP_RUN_ID  = run.info.run_id
    MODEL_VER   = info.registered_model_version

print(f"✅ Logged run        : {APP_RUN_ID}")
print(f"✅ UC model version  : {REGISTERED_MODEL}/{MODEL_VER}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 6 — Deploy to Model Serving + Enable AI Gateway
# MAGIC
# MAGIC One SDK call deploys (or updates) the endpoint; a second call attaches the AI Gateway with PII input + safety output guardrails and turns the inference table on.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🚀 STEP 6 - SERVING ENDPOINT + AI GATEWAY + INFERENCE TABLE
# ============================================================================

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AiGatewayConfig,
    AiGatewayGuardrails,
    AiGatewayGuardrailParameters,
    AiGatewayGuardrailPiiBehavior,
    AiGatewayGuardrailPiiBehaviorBehavior,
    AiGatewayInferenceTableConfig,
)

w = WorkspaceClient()

served_entity = ServedEntityInput(
    entity_name=REGISTERED_MODEL,
    entity_version=str(MODEL_VER),
    workload_size="Small",
    scale_to_zero_enabled=True,
)

endpoints = {e.name for e in w.serving_endpoints.list()}
config = EndpointCoreConfigInput(name=SERVING_ENDPOINT, served_entities=[served_entity])

if SERVING_ENDPOINT in endpoints:
    print(f"♻️  Updating existing endpoint {SERVING_ENDPOINT}…")
    w.serving_endpoints.update_config(name=SERVING_ENDPOINT, served_entities=[served_entity])
else:
    print(f"🆕 Creating endpoint {SERVING_ENDPOINT}…")
    w.serving_endpoints.create(name=SERVING_ENDPOINT, config=config)

w.serving_endpoints.wait_get_serving_endpoint_not_updating(name=SERVING_ENDPOINT)

# --- Attach AI Gateway (idempotent) ---
w.serving_endpoints.put_ai_gateway(
    name=SERVING_ENDPOINT,
    guardrails=AiGatewayGuardrails(
        input=AiGatewayGuardrailParameters(
            pii=AiGatewayGuardrailPiiBehavior(
                behavior=AiGatewayGuardrailPiiBehaviorBehavior.BLOCK
            ),
        ),
        output=AiGatewayGuardrailParameters(safety=True),
    ),
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        table_name_prefix=INFERENCE_PREFIX,
    ),
)

print(f"✅ Endpoint {SERVING_ENDPOINT} is live with AI Gateway and inference table")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 7 — Smoke Test: Clean + PII Query
# MAGIC
# MAGIC We hit the endpoint twice — once with a normal governance question, once with embedded PII. The first should return a grounded answer; the second should be **blocked at the gateway** before reaching the model.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🧪 STEP 7 - SMOKE TEST CLEAN + PII QUERIES
# ============================================================================

oai = w.serving_endpoints.get_open_ai_client()

clean_resp = oai.chat.completions.create(
    model=SERVING_ENDPOINT,
    messages=[{"role": "user", "content": "What is a Unity Catalog dynamic view?"}],
)
print("✅ Clean query passed:")
print(clean_resp.choices[0].message.content[:400], "\n")

PII_QUERY = (
    "My SSN is 123-45-6789 and email jane.doe@example.com — "
    "can you store these in a Unity Catalog table for me?"
)
try:
    blocked = oai.chat.completions.create(
        model=SERVING_ENDPOINT,
        messages=[{"role": "user", "content": PII_QUERY}],
    )
    print("⚠️  Unexpected: PII query was NOT blocked.")
    print(blocked.choices[0].message.content[:300])
except Exception as e:
    print("✅ PII query blocked at the gateway as expected.")
    print(f"   {e!r}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 8 — Verify the Inference Table Is Recording Traffic
# MAGIC
# MAGIC Inference table writes can lag a minute or two. Re-run the cell if the table is empty on first try; the rest of Module 8 depends on the table existing and capturing requests.
# MAGIC

# COMMAND ----------

# ============================================================================
# 🗃️ STEP 8 - VERIFY INFERENCE TABLE
# ============================================================================

print(f"Inference table: {INFERENCE_TABLE}")

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

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 9 — Persist Capstone Identifiers for Notebooks 2-6
# MAGIC
# MAGIC Other capstone notebooks read these constants. We log them as MLflow params for audit, and emit job task values so a Workflow can chain notebooks without copy-paste.
# MAGIC

# COMMAND ----------

# ============================================================================
# 📝 STEP 9 - PERSIST CAPSTONE IDENTIFIERS
# ============================================================================

CAPSTONE = {
    "catalog":           CATALOG,
    "schema":            SCHEMA,
    "docs_table":        DOCS_TABLE,
    "vs_endpoint":       VS_ENDPOINT,
    "vs_index":          VS_INDEX_FQN,
    "registered_model":  REGISTERED_MODEL,
    "model_version":     str(MODEL_VER),
    "serving_endpoint":  SERVING_ENDPOINT,
    "inference_table":   INFERENCE_TABLE,
    "experiment_path":   EXPERIMENT_PATH,
}

with mlflow.start_run(run_id=APP_RUN_ID):
    mlflow.log_dict(CAPSTONE, "capstone_constants.json")
    mlflow.log_params(CAPSTONE)

try:
    for k, v in CAPSTONE.items():
        dbutils.jobs.taskValues.set(key=k, value=v)
    print("✅ taskValues set for downstream Workflow tasks.")
except Exception:
    print("ℹ️  Not running in a Workflow — taskValues skipped (interactive mode).")

print("\nCapstone constants:")
for k, v in CAPSTONE.items():
    print(f"  {k:18s} = {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Lab Complete
# MAGIC
# MAGIC | Check | Status |
# MAGIC | --- | --- |
# MAGIC | UC governance corpus written to Delta with CDF on | ✅ |
# MAGIC | Vector Search endpoint + index online | ✅ |
# MAGIC | Traced RAG agent — RETRIEVER + LLM spans visible in MLflow | ✅ |
# MAGIC | Pyfunc model registered to Unity Catalog | ✅ |
# MAGIC | Model Serving endpoint deployed | ✅ |
# MAGIC | AI Gateway: PII input + safety output guardrails on | ✅ |
# MAGIC | Inference table provisioned and capturing requests | ✅ |
# MAGIC | Capstone constants persisted (MLflow + taskValues) | ✅ |
# MAGIC
# MAGIC **Next:** *Capstone 8.2* — assemble the **hybrid eval dataset** (synthetic + production traces + hand-curated edge cases) and save it as a UC-backed `mlflow.genai.datasets` resource.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📝 Summary
# MAGIC
# MAGIC In this notebook, we built the **App layer** of the capstone:
# MAGIC
# MAGIC ### 1. The App Is a Reproducible Pyfunc
# MAGIC - Wrapping the agent as `mlflow.pyfunc.PythonModel` is what makes it servable, versioned, and governed.
# MAGIC - `model_config` carries the pointers to VS endpoint/index — re-deployable into staging or prod by changing config alone.
# MAGIC
# MAGIC ### 2. AI Gateway Is the First Quality Layer
# MAGIC - Guardrails reject PII before the model runs — cheaper, faster, auditable.
# MAGIC - The inference table that the gateway writes is the **single source of truth** for Notebooks 2 (dataset bootstrap) and 5 (production monitoring).
# MAGIC
# MAGIC ### 3. Capstone Constants Are Contract
# MAGIC - All downstream notebooks read the same `CATALOG.SCHEMA` and endpoint names. Pinning them once eliminates drift between the gate and the running app.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Move to **Capstone 8.2** to build the eval dataset that Notebooks 3, 4, and 6 will all run against.
# MAGIC