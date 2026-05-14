import os
import streamlit as st
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain.llms.fake import FakeListLLM
from langchain_community.embeddings import FakeEmbeddings
import concurrent.futures

st.set_page_config(page_title="Intelligent Banking Support Analytics", layout="wide")

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("IntelligentBankingSupportAnalytics_UI") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_resource
def load_ml_model():
    return PipelineModel.load("spark_rf_model")

@st.cache_resource
def load_rag_pipeline():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    else:
        embeddings = FakeEmbeddings(size=768)
        llm = FakeListLLM(responses=["[Simulated Response] Based on the context, please dispatch a technician and adhere to the SLA compliance requirements."])
        
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return llm, vector_store

def invoke_support_agent(ticket_query, vector_store, llm):
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
        input_variables=["context", "issue"]
    )
    
    final_response_out = llm.invoke(synthesis_prompt.format(context=context, issue=ticket_query))
    final_response = final_response_out.content if hasattr(final_response_out, 'content') else final_response_out
    
    return final_response

st.markdown('<div class="main-header">Banking Operations Analytics Platform</div>', unsafe_allow_html=True)

try:
    spark = init_spark()
    model = load_ml_model()
    llm, vector_store = load_rag_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"System Initialization Error: {e}. Ensure the training pipeline (`python main.py`) has been executed first.")
    model_loaded = False

if model_loaded:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="section-header">Ticket Input Form</div>', unsafe_allow_html=True)
        
        contact_type = st.selectbox("Contact Type", ['ATM Alert', 'Phone', 'Mobile App', 'Branch Escalation', 'Internal Portal', 'Email'])
        category = st.selectbox("Category", ['ATM Services', 'Credit Card Operations', 'Loan Management System', 'Regulatory Reporting', 'Mobile Banking', 'Payment Processing', 'Fraud Detection System', 'Core Banking System', 'Internet Banking', 'Customer Data Management'])
        impact = st.selectbox("Impact", ["Low", "Medium", "High"])
        urgency = st.selectbox("Urgency", ["Low", "Medium", "High"])
        regulatory_risk = st.selectbox("Regulatory Risk", ["No", "Yes"])
        
        affected_customers = st.number_input("Affected Customers", min_value=0, max_value=1000000, value=1)
        financial_impact = st.number_input("Financial Impact ($)", min_value=0.0, max_value=100000000.0, value=0.0)
        
        ticket_text = st.text_area("Ticket Text Description (Short Description)", "ATM showing out of service but vault has sufficient cash.")
        
        analyze_btn = st.button("Analyze Banking Ticket")
        
    with col2:
        st.markdown('<div class="section-header">Analysis Output</div>', unsafe_allow_html=True)
        
        if analyze_btn and ticket_text:
            with st.spinner("Processing through ML pipeline..."):
                schema = StructType([
                    StructField("number", StringType(), True),
                    StructField("sys_created_on", TimestampType(), True),
                    StructField("contact_type", StringType(), True),
                    StructField("priority", StringType(), True), # Dummy
                    StructField("short_description", StringType(), True),
                    StructField("service_offering", StringType(), True), # Dummy
                    StructField("state", StringType(), True), # Dummy
                    StructField("close_code", StringType(), True), # Dummy
                    StructField("close_notes", StringType(), True), # Dummy
                    StructField("assigned_to", StringType(), True), # Dummy
                    StructField("category", StringType(), True),
                    StructField("subcategory", StringType(), True), # Dummy
                    StructField("impact", StringType(), True),
                    StructField("urgency", StringType(), True),
                    StructField("resolution_time_hours", DoubleType(), True), # Dummy
                    StructField("affected_customers", IntegerType(), True),
                    StructField("financial_impact", DoubleType(), True),
                    StructField("regulatory_risk", StringType(), True)
                ])
                
                input_data = [("INC_TEST", datetime.now(), contact_type, "Low", 
                              ticket_text, category, "New", "N/A", "N/A", "System", 
                              category, "Subcat", impact, urgency, 0.0, 
                              int(affected_customers), float(financial_impact), regulatory_risk)]
                
                input_df = spark.createDataFrame(input_data, schema)
                
                # Predict Priority
                pred_df = model.transform(input_df)
                predicted_priority = pred_df.select("predicted_priority").collect()[0][0]
                
                # Generate Answer via Single Agent
                final_answer = invoke_support_agent(
                    ticket_text, vector_store, llm
                )
                
                st.markdown(f'''
                <div class="result-box">
                    <p style="margin-bottom: 5px; color: #6b7280; font-size: 14px;">Predicted Priority Level</p>
                    <h3 class="priority-{predicted_priority}" style="margin-top: 0;">{predicted_priority} Priority</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="result-box">
                    <p style="margin-bottom: 5px; color: #6b7280; font-size: 14px;">Synthesized Support Resolution</p>
                    <p style="margin-top: 0; color: #1f2937; font-size: 16px; line-height: 1.5;">{final_answer}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                st.info("System Note: Model inference and knowledge base retrieval were executed successfully.")
