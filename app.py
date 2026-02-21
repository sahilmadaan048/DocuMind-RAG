import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import pandas as pd
from docx import Document as DocxDocument

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from pinecone import Pinecone, ServerlessSpec
import os


# LOAD ENVIRONMENT VARIABLES
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# DOCX extraction
def extract_docx_documents(docx_files):
    documents = []

    for file in docx_files:
        doc = DocxDocument(file)
        full_text = []

        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)

        content = "\n".join(full_text)

        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file.name,
                        "page": 1  # DOCX treated as single page
                    }
                )
            )

    return documents


# CSV extraction
def extract_csv_documents(csv_files):
    documents = []

    for file in csv_files:
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            row_text = ", ".join(
                f"{col}: {row[col]}" for col in df.columns
            )

            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source": file.name,
                        "page": index + 1  # row number
                    }
                )
            )

    return documents


# PDF DOCUMENT EXTRACTION WITH METADATA (SOURCE + PAGE)
def extract_pdf_documents(pdf_files):
    documents = []

    for pdf in pdf_files:
        reader = PdfReader(pdf)

        for page_number, page in enumerate(reader.pages):
            content = page.extract_text()

            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": pdf.name,
                            "page": page_number + 1
                        }
                    )
                )

    return documents


# DOCUMENT CHUNKING (PRESERVES METADATA)
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)


# VECTOR STORE CREATION (FAISS)
def create_vector_store(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    vector_store.add_documents(chunks)


# LOAD LCEL RAG CHAIN (RETURNS ANSWER + SOURCES)
from langchain_core.runnables import RunnableMap

def load_rag_chain(selected_model):

    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    db = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )
    # Semantic Retriever
    vector_retriever = db.as_retriever(search_kwargs={"k": 4})

    # BM25 Keyword Retriever
    docs = list(db.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4

    # Hybrid Retriever (Weighted)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    # Dynamic LLM (Multi-Model)
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
        streaming=True
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering questions strictly based on provided context.

If the answer is not found in the context, respond with:
"Answer not available in the provided documents."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Step 1: Retrieve Context
    retrieval_chain = RunnableMap({
        "context": hybrid_retriever,
        "question": RunnablePassthrough()
    })

    # Step 2: Generate Answer
    answer_chain = (
        RunnableMap({
            "context": lambda x: format_docs(x["context"]),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 3: Combine Answer + Sources
    final_chain = retrieval_chain.assign(
        answer=answer_chain,
        sources=lambda x: x["context"]
    )

    return final_chain


# STREAMLIT UI
def main():

    st.set_page_config(
        page_title="DocuMind-RAG",
        page_icon="📄"
    )

    st.title("DocuMind-RAG")
    st.write("Multi-Document Retrieval Augmented Generation System")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # USER QUESTION INPUT
    question = st.text_input("Ask a question about your documents")

    if question.strip() and st.session_state.rag_chain:

        result = st.session_state.rag_chain.invoke(question)

        st.write("### Response")
        st.write(result["answer"])

        # Display Sources
        st.write("### Sources")

        unique_sources = set()

        for doc in result["sources"]:
            source_info = f"{doc.metadata['source']} (Page {doc.metadata['page']})"

            if source_info not in unique_sources:
                st.write(f"- 📄 {source_info}")
                unique_sources.add(source_info)

    # SIDEBAR
    with st.sidebar:

        # Model Selection
        st.header("Model Settings")

        available_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash-001"
        ]

        selected_model = st.selectbox(
            "Select LLM Model",
            available_models
        )

        # Track model change
        if "current_model" not in st.session_state:
            st.session_state.current_model = selected_model

        if st.session_state.current_model != selected_model:
            st.session_state.current_model = selected_model
            if os.path.exists("faiss_index"):
                st.session_state.rag_chain = load_rag_chain(selected_model)

        # Document Upload
        st.header("Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, DOCX, CSV)",
            type=["pdf", "docx", "csv"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):

            if uploaded_files:

                with st.spinner("Processing Documents..."):

                    documents = []

                    pdf_files = [f for f in uploaded_files if f.name.endswith(".pdf")]
                    docx_files = [f for f in uploaded_files if f.name.endswith(".docx")]
                    csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]

                    if pdf_files:
                        documents.extend(extract_pdf_documents(pdf_files))

                    if docx_files:
                        documents.extend(extract_docx_documents(docx_files))

                    if csv_files:
                        documents.extend(extract_csv_documents(csv_files))

                    chunks = split_documents(documents)
                    create_vector_store(chunks)

                    st.session_state.rag_chain = load_rag_chain(selected_model)

                st.success("Documents processed successfully.")

            else:
                st.warning("Please upload at least one document.")


# RUN APPLICATION
if __name__ == "__main__":
    main()