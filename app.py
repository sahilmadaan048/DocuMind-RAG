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
from langchain_core.documents import Document


# LOAD ENVIRONMENT VARIABLES
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


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

    vector_store = FAISS.from_documents(
        chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# LOAD LCEL RAG CHAIN (RETURNS ANSWER + SOURCES)
from langchain_core.runnables import RunnableMap


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

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        streaming=True
    )

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

    # Step 1: Retrieve context
    retrieval_chain = RunnableMap({
        "context": retriever,
        "question": RunnablePassthrough()
    })

    # Step 2: Generate answer
    answer_chain = (
        RunnableMap({
            "context": lambda x: format_docs(x["context"]),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 3: Combine answer + sources
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

    # PROCESS QUESTION
    if question and st.session_state.rag_chain:

        result = st.session_state.rag_chain.invoke(question)

        st.write("### Response")
        st.write(result["answer"])

        # DISPLAY SOURCE CITATIONS
        st.write("### Sources")

        unique_sources = set()

        for doc in result["sources"]:
            source_info = f"{doc.metadata['source']} (Page {doc.metadata['page']})"

            if source_info not in unique_sources:
                st.write(f"- 📄 {source_info}")
                unique_sources.add(source_info)

    # SIDEBAR
    with st.sidebar:

        st.header("Upload Documents")

        pdf_files = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True
        )

        if st.button("Process Documents"):

            if pdf_files:

                with st.spinner("Processing Documents..."):

                    documents = extract_pdf_documents(pdf_files)
                    chunks = split_documents(documents)
                    create_vector_store(chunks)

                    st.session_state.rag_chain = load_rag_chain()

                st.success("Documents processed successfully.")

            else:
                st.warning("Please upload at least one PDF.")


# RUN APPLICATION
if __name__ == "__main__":
    main()
