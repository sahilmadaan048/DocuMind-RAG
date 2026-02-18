import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# PDF TEXT EXTRACTION
def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


# TEXT SPLITTING
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# VECTOR STORE CREATION
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# LOAD LCEL RAG CHAIN (Streaming Enabled)
def load_rag_chain():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    # 🔥 Streaming Enabled Here
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering questions based strictly on provided context.

If the answer is not found in the context, respond with:
"Answer not available in the provided documents."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


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

    # User input
    question = st.text_input("Ask a question about your documents")

    if question and st.session_state.rag_chain:

        st.write("### Response")

        response_container = st.empty()
        full_response = ""

        # Streaming response
        for chunk in st.session_state.rag_chain.stream(question):
            full_response += chunk
            response_container.markdown(full_response)

    # Sidebar
    with st.sidebar:
        st.header("Upload Documents")

        pdf_files = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if pdf_files:
                with st.spinner("Processing Documents..."):
                    raw_text = extract_pdf_text(pdf_files)
                    chunks = split_text(raw_text)
                    create_vector_store(chunks)
                    st.session_state.rag_chain = load_rag_chain()

                st.success("Documents processed successfully.")
            else:
                st.warning("Please upload at least one PDF.")


if __name__ == "__main__":
    main()
