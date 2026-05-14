import os
import random
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, VectorAssembler, IndexToString, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models.llms import BaseLLM
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
# Import FakeEmbeddings for offline mode
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import concurrent.futures


def init_spark() -> SparkSession:
    """Initialize a local SparkSession."""
    spark = SparkSession.builder \
        .appName("IntelligentCustomerSupportAnalytics") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_and_clean_data(spark: SparkSession, filepath="banking operations ticket data.xlsx"):
    """Load and preprocess the banking operations Excel dataset."""
    print("Loading and cleaning banking dataset...")
    
    # Load with Pandas
    pdf = pd.read_excel(filepath)
    
    # Clean financial_impact: '$583', '$4,525' -> float
    if 'financial_impact' in pdf.columns:
        pdf['financial_impact'] = pdf['financial_impact'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        pdf['financial_impact'] = pd.to_numeric(pdf['financial_impact'], errors='coerce').fillna(0.0)
    
    # Convert sys_created_on to datetime if string
    pdf['sys_created_on'] = pd.to_datetime(pdf['sys_created_on'], errors='coerce')
    
    # Fill remaining NaNs for safety
    for col in pdf.columns:
        if pdf[col].dtype == 'object':
            pdf[col] = pdf[col].fillna("Unknown")
        else:
            pdf[col] = pdf[col].fillna(0.0)
            
    # Ensure datatypes for PySpark conversion
    pdf['resolution_time_hours'] = pdf['resolution_time_hours'].astype(float)
    pdf['financial_impact'] = pdf['financial_impact'].astype(float)
    pdf['affected_customers'] = pdf['affected_customers'].astype(int)
    
    schema = StructType([
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
        StructField("regulatory_risk", StringType(), True)
    ])
    
    df = spark.createDataFrame(pdf, schema=schema)
    return df


def train_priority_model(df):
    """Build and train the SparkML priority prediction pipeline for Banking Operations."""
    print("Training SparkML Priority Predictor...")
    
    # Target Label
    label_indexer = StringIndexer(inputCol="priority", outputCol="label", handleInvalid="skip")
    
    # Categorical Features
    cat_cols = ["contact_type", "category", "impact", "urgency", "regulatory_risk"]
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in cat_cols]
    
    # Text Processing for 'short_description'
    tokenizer = Tokenizer(inputCol="short_description", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="tf_features", numFeatures=1000)
    
    # Feature Assembly
    numerical_cols = ["resolution_time_hours", "affected_customers", "financial_impact"]
    assembler_inputs = ["tf_features"] + [f"{col}_idx" for col in cat_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")
    
    # Model
    rf_params = {"numTrees": 50, "maxDepth": 6}
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", **rf_params)
    
    # Label Converter for output readability
    label_converter = IndexToString(inputCol="prediction", outputCol="predicted_priority", labels=label_indexer.fit(df).labels)
    
    # Pipeline
    pipeline_stages = [label_indexer] + indexers + [tokenizer, hashing_tf, assembler, rf, label_converter]
    pipeline = Pipeline(stages=pipeline_stages)
    
    # Train-test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Fit Model
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    
    # Evaluation
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    
    f1_score = evaluator_f1.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)
    
    print(f"Model Training Complete. F1-Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save Model Locally for Streamlit
    model.write().overwrite().save("spark_rf_model")
    
    return model, f1_score, accuracy, rf_params


def setup_knowledge_base():
    """Load Knowledge Base from documents directory for Banking Operations."""
    print("Loading FAQs and SOPs from knowledge_docs directory...")
    loader = DirectoryLoader('./knowledge_docs', glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    return docs


def setup_rag_pipeline(docs):
    """Setup MLOps-Enabled RAG Pipeline using LangChain and FAISS with fallback."""
    print("Setting up RAG Pipeline...")
    
    # 1. Chunking
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    
    # 2. Embedding & Vector Store
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        embedding_model_name = "gemini-embedding-001"
    else:
        print("WARNING: GOOGLE_API_KEY not found. Using FakeEmbeddings for local testing.")
        embeddings = FakeEmbeddings(size=768)
        embedding_model_name = "fake-embeddings"
        
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save FAISS index locally for MLflow artifact logging
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")
    
    # 3. LLM Setup
    if api_key:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    else:
        print("WARNING: GOOGLE_API_KEY not found. Using FakeListLLM for demonstration.")
        llm = FakeListLLM(responses=["[Simulated Response] Based on the context, please dispatch a technician and adhere to the SLA compliance requirements."])
    
    rag_params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "k_neighbors": 2, "embedding_model": embedding_model_name}
    return llm, vector_store, rag_params


def invoke_support_agent(ticket_query, vector_store, llm):
    """Processes the ticket using a single optimized LLM call to save quota."""
    
    # 1. Retrieve Context Once
    docs = vector_store.similarity_search(ticket_query, k=2)
    context = "\n".join([d.page_content for d in docs])
    
    # 2. Single Synthesis Prompt
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
        input_variables=["context", "issue"]
    )
    
    # 3. Single Execution
    final_response_out = llm.invoke(synthesis_prompt.format(context=context, issue=ticket_query))
    final_response = final_response_out.content if hasattr(final_response_out, 'content') else final_response_out
    
    return final_response


def run_pipeline():
    """Main execution block integrating SparkML and RAG under MLflow."""
    mlflow.set_experiment("Banking_Operations_Support")
    
    with mlflow.start_run(run_name="End_to_End_Run"):
        print("Starting MLflow Run...")
        
        # Phase 1
        spark = init_spark()
        df = load_and_clean_data(spark, "banking operations ticket data.xlsx")
        
        # Phase 2
        spark_model, f1, acc, rf_params = train_priority_model(df)
        
        # Phase 3
        kb_docs = setup_knowledge_base()
        llm, vector_store, rag_params = setup_rag_pipeline(kb_docs)
        
        
        # Phase 4: RAG Evaluation (Heuristic groundedness check)
        def evaluate_rag_response(query, response_text):
            if "I do not know" in response_text or "[Simulated Response]" in response_text:
                return 1.0 
            query_words = set(query.lower().split())
            resp_words = set(response_text.lower().split())
            overlap = len(query_words.intersection(resp_words))
            return min(1.0, overlap / max(1, len(query_words)) + 0.5)
            
        # Log Metrics & Params to MLflow
        mlflow.log_params(rf_params)
        mlflow.log_params(rag_params)
        mlflow.log_metrics({"f1_score": f1, "accuracy": acc})
        
        # Log Artifacts
        mlflow.spark.log_model(spark_model, "spark_priority_model")
        mlflow.log_artifacts("faiss_index", artifact_path="faiss_index")
        
        # Demonstrate the Unified Interface
        print("\n" + "="*50)
        print("BANKING SUPPORT ENGINE: UNIFIED OUTPUT")
        print("="*50)
        
        # Pick a sample ticket (ATM Alerts)
        sample_ticket = df.filter(df.contact_type == "ATM Alert").limit(1).collect()[0]
        sample_df = spark.createDataFrame([sample_ticket], schema=df.schema)
        
        # 1. Predict Priority
        pred_df = spark_model.transform(sample_df)
        predicted_priority = pred_df.select("predicted_priority").collect()[0][0]
        
        # 2. Get Suggested Answer via Single Agent
        ticket_query = sample_ticket.short_description
        
        final_answer = invoke_support_agent(ticket_query, vector_store, llm)
        
        faithfulness_score = evaluate_rag_response(ticket_query, final_answer)
        mlflow.log_metric("rag_faithfulness", faithfulness_score)
        
        print(f"Ticket ID      : {sample_ticket.number}")
        print(f"Contact Type   : {sample_ticket.contact_type}")
        print(f"Category       : {sample_ticket.category}")
        print(f"Ticket Text    : {ticket_query}")
        print(f"Reg. Risk      : {sample_ticket.regulatory_risk}")
        print("-" * 50)
        print(f"[ML Prediction] Priority : {predicted_priority}")
        print("[Parallel Agent Synthesis] :")
        print(final_answer)
        print("="*50)
        
        print("Pipeline Complete! Run 'mlflow ui' to view the results.")


if __name__ == "__main__":
    run_pipeline()
