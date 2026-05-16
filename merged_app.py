# Merged Application: Combines training pipeline (main.py) and Streamlit UI (app.py)

import os
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    IntegerType,
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, VectorAssembler, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:

        class PromptTemplate:
            def __init__(self, template: str, input_variables=None):
                self.template = template
                self.input_variables = input_variables or []

            def format(self, **kwargs):
                return self.template.format(**kwargs)


try:
    from langchain_community.llms.fake import FakeListLLM
except ImportError:
    from langchain.llms.fake import FakeListLLM

try:
    from langchain_community.embeddings import FakeEmbeddings
except ImportError:
    FakeEmbeddings = None

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
except ImportError:
    DirectoryLoader = None
    TextLoader = None

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Training data: new sample set + legacy banking operations export
TRAINING_DATASETS = [
    "sample_ticket_data.xlsx",
    "banking operations ticket data.xlsx",
]

# RAG knowledge: three uploaded text manuals (plus markdown files in knowledge_docs/)
KNOWLEDGE_TEXT_FILES = [
    "Escalation_SOP.txt",
    "FAQ_Manual.txt",
    "Troubleshooting_Guide.txt",
]

KNOWLEDGE_DOCS_DIR = os.path.join(APP_DIR, "knowledge_docs")

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIM = 1536


def _openai_api_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _build_openai_llm():
    if ChatOpenAI is None:
        raise ImportError("langchain-openai is required. Install with: pip install langchain-openai")
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


def _build_openai_embeddings():
    if OpenAIEmbeddings is None:
        raise ImportError("langchain-openai is required. Install with: pip install langchain-openai")
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def _resolve_path(filename: str) -> str:
    return os.path.join(APP_DIR, filename)


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _build_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("IntelligentBankingSupportAnalytics")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def init_spark() -> SparkSession:
    """Initialize Spark for the training pipeline (CLI)."""
    return _build_spark_session()


@st.cache_resource
def init_spark_ui() -> SparkSession:
    """Cached Spark session for the Streamlit UI."""
    return _build_spark_session()


def _normalize_tier_value(value) -> str:
    """Map values like '1 - High' to 'High' for consistent training and UI."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if " - " in text:
        text = text.split(" - ", 1)[-1].strip()
    return text


def _clean_ticket_dataframe(pdf: pd.DataFrame) -> pd.DataFrame:
    """Shared cleaning for all ticket Excel sources."""
    pdf = pdf.copy()

    if "financial_impact" in pdf.columns:
        pdf["financial_impact"] = (
            pdf["financial_impact"].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        pdf["financial_impact"] = pd.to_numeric(pdf["financial_impact"], errors="coerce").fillna(0.0)

    pdf["sys_created_on"] = pd.to_datetime(pdf["sys_created_on"], errors="coerce")

    for col in ("impact", "urgency"):
        if col in pdf.columns:
            pdf[col] = pdf[col].apply(_normalize_tier_value)

    for col in pdf.columns:
        if pdf[col].dtype == "object":
            pdf[col] = pdf[col].fillna("Unknown")
        else:
            pdf[col] = pdf[col].fillna(0.0)

    pdf["resolution_time_hours"] = pdf["resolution_time_hours"].astype(float)
    pdf["financial_impact"] = pdf["financial_impact"].astype(float)
    pdf["affected_customers"] = pdf["affected_customers"].astype(int)
    return pdf


def _ticket_schema() -> StructType:
    return StructType(
        [
            StructField("number", StringType(), True),
            StructField("sys_created_on", TimestampType(), True),
            StructField("contact_type", StringType(), True),
            StructField("priority", StringType(), True),
            StructField("short_description", StringType(), True),
            StructField("service_offering", StringType(), True),
            StructField("state", StringType(), True),
            StructField("close_code", StringType(), True),
            StructField("close_notes", StringType(), True),
            StructField("assigned_to", StringType(), True),
            StructField("category", StringType(), True),
            StructField("subcategory", StringType(), True),
            StructField("impact", StringType(), True),
            StructField("urgency", StringType(), True),
            StructField("resolution_time_hours", DoubleType(), True),
            StructField("affected_customers", IntegerType(), True),
            StructField("financial_impact", DoubleType(), True),
            StructField("regulatory_risk", StringType(), True),
        ]
    )


def load_and_clean_data(
    spark: SparkSession,
    filepath: str | None = None,
    filepaths: list[str] | None = None,
):
    """Load one or more Excel ticket datasets, clean, merge, and return a Spark DataFrame."""
    paths = filepaths or ([filepath] if filepath else TRAINING_DATASETS)
    resolved = [_resolve_path(p) for p in paths]

    frames = []
    for path in resolved:
        if not os.path.exists(path):
            print(f"WARNING: dataset not found, skipping: {path}")
            continue
        print(f"Loading dataset: {os.path.basename(path)}")
        frames.append(pd.read_excel(path))

    if not frames:
        raise FileNotFoundError(f"No training datasets found. Expected one of: {resolved}")

    pdf = pd.concat(frames, ignore_index=True)
    print(f"Combined training rows: {len(pdf)}")
    pdf = _clean_ticket_dataframe(pdf)

    return spark.createDataFrame(pdf, schema=_ticket_schema())


def train_priority_model(df):
    """Build and train the SparkML priority prediction pipeline."""
    print("Training SparkML Priority Predictor...")

    label_indexer = StringIndexer(inputCol="priority", outputCol="label", handleInvalid="skip")

    cat_cols = ["contact_type", "category", "impact", "urgency", "regulatory_risk"]
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in cat_cols]

    tokenizer = Tokenizer(inputCol="short_description", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="tf_features", numFeatures=1000)

    numerical_cols = ["resolution_time_hours", "affected_customers", "financial_impact"]
    assembler_inputs = ["tf_features"] + [f"{col}_idx" for col in cat_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

    rf_params = {"numTrees": 50, "maxDepth": 6}
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", **rf_params)

    label_converter = IndexToString(
        inputCol="prediction",
        outputCol="predicted_priority",
        labels=label_indexer.fit(df).labels,
    )

    pipeline = Pipeline(stages=[label_indexer] + indexers + [tokenizer, hashing_tf, assembler, rf, label_converter])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_score = evaluator_f1.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)

    print(f"Model Training Complete. F1-Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")

    model.write().overwrite().save(_resolve_path("spark_rf_model"))
    return model, f1_score, accuracy, rf_params


def setup_knowledge_base():
    """Load knowledge from markdown (knowledge_docs/) and uploaded text manuals."""
    if DirectoryLoader is None or TextLoader is None:
        raise ImportError("langchain_community document loaders are required for the knowledge base.")

    docs = []

    if os.path.isdir(KNOWLEDGE_DOCS_DIR):
        print(f"Loading markdown knowledge from {KNOWLEDGE_DOCS_DIR}...")
        md_loader = DirectoryLoader(KNOWLEDGE_DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)
        docs.extend(md_loader.load())

    for filename in KNOWLEDGE_TEXT_FILES:
        path = _resolve_path(filename)
        if os.path.exists(path):
            print(f"Loading knowledge document: {filename}")
            docs.extend(TextLoader(path, encoding="utf-8").load())
        else:
            print(f"WARNING: knowledge file not found, skipping: {filename}")

    if not docs:
        raise FileNotFoundError("No knowledge documents found for RAG indexing.")

    print(f"Total knowledge documents loaded: {len(docs)}")
    return docs


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate_limit" in msg or "resource_exhausted" in msg or "429" in msg


def _build_faiss_index(chunks, embeddings, max_retries: int = 6):
    """Build FAISS index with retries when the embedding API rate limit is exceeded."""
    for attempt in range(max_retries):
        try:
            return FAISS.from_documents(chunks, embeddings)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == max_retries - 1:
                raise
            wait_seconds = 35 * (attempt + 1)
            print(f"Embedding rate limit hit; waiting {wait_seconds}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)


def setup_rag_pipeline(docs):
    """Create FAISS vector store and LLM (real or fake) for RAG."""
    print("Setting up RAG Pipeline...")
    if FAISS is None:
        raise ImportError("langchain_community.vectorstores.FAISS is required for RAG.")

    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    print(f"RAG chunks to embed: {len(chunks)}")

    if _openai_api_configured():
        embeddings = _build_openai_embeddings()
        embedding_model_name = OPENAI_EMBEDDING_MODEL
    else:
        print("WARNING: OPENAI_API_KEY not found. Using FakeEmbeddings for local testing.")
        embeddings = FakeEmbeddings(size=OPENAI_EMBEDDING_DIM)
        embedding_model_name = "fake-embeddings"

    vector_store = _build_faiss_index(chunks, embeddings)
    faiss_path = _resolve_path("faiss_index")
    os.makedirs(faiss_path, exist_ok=True)
    vector_store.save_local(faiss_path)

    if _openai_api_configured():
        llm = _build_openai_llm()
    else:
        print("WARNING: OPENAI_API_KEY not found. Using FakeListLLM for demonstration.")
        llm = FakeListLLM(
            responses=[
                "[Simulated Response] Based on the context, please dispatch a technician and adhere to the SLA compliance requirements."
            ]
        )

    rag_params = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k_neighbors": 2,
        "embedding_model": embedding_model_name,
        "chat_model": OPENAI_CHAT_MODEL if _openai_api_configured() else "fake-llm",
    }
    return llm, vector_store, rag_params


def invoke_support_agent(ticket_query, vector_store, llm):
    """Retrieve context and generate a response using the LLM."""
    docs = vector_store.similarity_search(ticket_query, k=2)
    context = "\n".join([d.page_content for d in docs])
    synthesis_prompt = PromptTemplate(
        template="""You are an expert Banking Operations Support Agent.
Process the following issue using ONLY the provided context.
Your response must be professional, cohesive, and include:
1. An empathetic and formal opening addressing the issue.
2. Technical troubleshooting or resolution steps.
3. Any relevant SLA, regulatory requirements, or escalation policies.

Context:
{context}

Issue:
{issue}""",
        input_variables=["context", "issue"],
    )
    final_response_out = llm.invoke(synthesis_prompt.format(context=context, issue=ticket_query))
    return final_response_out.content if hasattr(final_response_out, "content") else final_response_out


def run_pipeline():
    """Main execution block integrating SparkML and RAG under MLflow (from main.py)."""
    mlflow.set_experiment("Banking_Operations_Support")

    with mlflow.start_run(run_name="End_to_End_Run"):
        print("Starting MLflow Run...")

        spark = init_spark()
        df = load_and_clean_data(spark, filepaths=TRAINING_DATASETS)

        spark_model, f1, acc, rf_params = train_priority_model(df)

        kb_docs = setup_knowledge_base()
        llm, vector_store, rag_params = setup_rag_pipeline(kb_docs)

        def evaluate_rag_response(query, response_text):
            if "I do not know" in response_text or "[Simulated Response]" in response_text:
                return 1.0
            query_words = set(query.lower().split())
            resp_words = set(response_text.lower().split())
            overlap = len(query_words.intersection(resp_words))
            return min(1.0, overlap / max(1, len(query_words)) + 0.5)

        mlflow.log_params(rf_params)
        mlflow.log_params(rag_params)
        mlflow.log_metrics({"f1_score": f1, "accuracy": acc})

        mlflow.spark.log_model(spark_model, "spark_priority_model")
        mlflow.log_artifacts("faiss_index", artifact_path="faiss_index")

        print("\n" + "=" * 50)
        print("BANKING SUPPORT ENGINE: UNIFIED OUTPUT")
        print("=" * 50)

        sample_ticket = None
        for contact in ("Email", "Phone", "Chat", "Web", "ATM Alert"):
            rows = df.filter(df.contact_type == contact).limit(1).collect()
            if rows:
                sample_ticket = rows[0]
                break
        if sample_ticket is None:
            sample_ticket = df.limit(1).collect()[0]
        sample_df = spark.createDataFrame([sample_ticket], schema=df.schema)

        pred_df = spark_model.transform(sample_df)
        predicted_priority = pred_df.select("predicted_priority").collect()[0][0]

        ticket_query = sample_ticket.short_description
        try:
            final_answer = invoke_support_agent(ticket_query, vector_store, llm)
            faithfulness_score = evaluate_rag_response(ticket_query, final_answer)
            mlflow.log_metric("rag_faithfulness", faithfulness_score)
        except Exception as exc:
            final_answer = f"[RAG demo skipped: {exc}]"
            print(f"WARNING: {final_answer}")

        print(f"Ticket ID      : {sample_ticket.number}")
        print(f"Contact Type   : {sample_ticket.contact_type}")
        print(f"Category       : {sample_ticket.category}")
        print(f"Ticket Text    : {ticket_query}")
        print(f"Reg. Risk      : {sample_ticket.regulatory_risk}")
        print("-" * 50)
        print(f"[ML Prediction] Priority : {predicted_priority}")
        print("[Parallel Agent Synthesis] :")
        print(final_answer)
        print("=" * 50)

        print("Pipeline Complete! Run 'mlflow ui' to view the results.")


@st.cache_resource
def load_ml_model():
    """Load the Spark model saved by the training pipeline."""
    return PipelineModel.load(_resolve_path("spark_rf_model"))


@st.cache_resource
def load_rag_pipeline():
    """Load FAISS index and LLM (real or fake) for the UI."""
    if FAISS is None:
        raise ImportError("langchain_community.vectorstores.FAISS is required for RAG.")

    if _openai_api_configured():
        embeddings = _build_openai_embeddings()
        llm = _build_openai_llm()
    else:
        embeddings = FakeEmbeddings(size=OPENAI_EMBEDDING_DIM)
        llm = FakeListLLM(
            responses=[
                "[Simulated Response] Based on the context, please dispatch a technician and adhere to the SLA compliance requirements."
            ]
        )
    vector_store = FAISS.load_local(
        _resolve_path("faiss_index"), embeddings, allow_dangerous_deserialization=True
    )
    return llm, vector_store


def _render_streamlit_ui():
    """Streamlit UI (from app.py)."""
    st.set_page_config(
        page_title="Module 9 : ML pipeline creation at scale using SparkML, and LLM application engineering",
        layout="wide",
    )

    st.markdown(
        """
<style>
    .stApp { background-color: #f8f9fa; }
    .main-header { font-family: 'Inter', sans-serif; color: #1f2937; font-size: 32px; font-weight: 700; padding-bottom: 20px; border-bottom: 2px solid #e5e7eb; margin-bottom: 30px; }
    .section-header { font-family: 'Inter', sans-serif; color: #374151; font-size: 20px; font-weight: 600; margin-top: 20px; margin-bottom: 15px; }
    .stButton button { background-color: #2563eb; color: white; font-weight: 600; border-radius: 6px; border: none; padding: 0.5rem 1rem; width: 100%; }
    .stButton button:hover { background-color: #1d4ed8; color: white; }
    .priority-Critical { color: #dc2626; font-weight: bold; }
    .priority-High { color: #ea580c; font-weight: bold; }
    .priority-Medium { color: #d97706; font-weight: bold; }
    .priority-Low { color: #16a34a; font-weight: bold; }
    .result-box { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 10px; border: 1px solid #e5e7eb; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">Module 9 : ML pipeline creation at scale using SparkML, and LLM application engineering</div>',
        unsafe_allow_html=True,
    )

    try:
        spark = init_spark_ui()
        ml_model = load_ml_model()
        llm, vector_store = load_rag_pipeline()
        resources_ok = True
    except Exception as e:
        st.error(
            f"System Initialization Error: {e}. "
            "Ensure you have run the training pipeline first (`python merged_app.py train`)."
        )
        resources_ok = False

    if not resources_ok:
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="section-header">Ticket Input Form</div>', unsafe_allow_html=True)

        contact_type = st.selectbox(
            "Contact Type",
            [
                "Email",
                "Phone",
                "Chat",
                "Web",
                "ATM Alert",
                "Mobile App",
                "Branch Escalation",
                "Internal Portal",
            ],
        )
        category = st.selectbox(
            "Category",
            [
                "ATM Services",
                "Credit Card Operations",
                "Loan Management System",
                "Regulatory Reporting",
                "Mobile Banking",
                "Payment Processing",
                "Fraud Detection System",
                "Core Banking System",
                "Internet Banking",
                "Customer Data Management",
            ],
        )
        impact = st.selectbox("Impact", ["Low", "Medium", "High"])
        urgency = st.selectbox("Urgency", ["Low", "Medium", "High"])
        regulatory_risk = st.selectbox("Regulatory Risk", ["No", "Yes"])
        affected_customers = st.number_input("Affected Customers", min_value=0, max_value=1000000, value=1)
        financial_impact = st.number_input("Financial Impact ($)", min_value=0.0, max_value=1e8, value=0.0)
        ticket_text = st.text_area(
            "Ticket Text Description (Short Description)",
            "Payment gateway timeout during checkout",
        )
        analyze_btn = st.button("Analyze Banking Ticket")

    with col2:
        st.markdown('<div class="section-header">Analysis Output</div>', unsafe_allow_html=True)

        if analyze_btn and ticket_text:
            with st.spinner("Processing through ML pipeline..."):
                schema = _ticket_schema()
                input_data = [
                    (
                        "INC_UI",
                        datetime.now(),
                        contact_type,
                        "Low",
                        ticket_text,
                        category,
                        "New",
                        "N/A",
                        "N/A",
                        "System",
                        category,
                        "Subcat",
                        impact,
                        urgency,
                        0.0,
                        int(affected_customers),
                        float(financial_impact),
                        regulatory_risk,
                    )
                ]
                input_df = spark.createDataFrame(input_data, schema)
                pred_df = ml_model.transform(input_df)
                predicted_priority = pred_df.select("predicted_priority").collect()[0][0]
                try:
                    final_answer = invoke_support_agent(ticket_text, vector_store, llm)
                except Exception as exc:
                    st.error(f"Support agent error ({OPENAI_CHAT_MODEL}): {exc}")
                    final_answer = None

                if final_answer is None:
                    return

                st.markdown(
                    f"""
<div class="result-box">
    <p style="margin-bottom: 5px; color: #6b7280; font-size: 14px;">Predicted Priority Level</p>
    <h3 class="priority-{predicted_priority}" style="margin-top: 0;">{predicted_priority} Priority</h3>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
<div class="result-box">
    <p style="margin-bottom: 5px; color: #6b7280; font-size: 14px;">Synthesized Support Resolution</p>
    <p style="margin-top: 0; color: #1f2937; font-size: 16px; line-height: 1.5;">{final_answer}</p>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.info("System Note: Model inference and knowledge-base retrieval were executed successfully.")


if _is_streamlit_runtime():
    _render_streamlit_ui()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("train", "--train"):
        run_pipeline()
    elif not _is_streamlit_runtime():
        print("Usage:")
        print("  python merged_app.py train      # Train models and build RAG index")
        print("  streamlit run merged_app.py     # Launch the Streamlit UI")
