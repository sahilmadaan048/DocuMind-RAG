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

# Load the env variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Extract text from the PDF's
def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content 

    return text

# split the text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    return splitter.split_text(text)

# create a vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )
    
    vector_store.save_local("faiss_index")

# load LCEF (langchain expression langcuage) RAG chain
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})  # return k most similar chunks

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering questions based strictly on provided context.

If the answer is not found in the context, respond with:
"Answer not available in the provided documents."

Context:
{context}

Question:
{question}
""")
    
    # format the retrived documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # LCEL RAG pipeline
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


# streamlit UI
def main():
    st.set_page_config(
        page_title="DocuMind-RAG",
        page_icon="📄"
    )

    st.title("DocuMind-RAG")
    st.write("Multi-Document Retrieval Augmented Generation System")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None 
    
    # user question input
    question = st.text_input("Ask a question about your documents")

    if question and st.session_state.rag_chain:
        response = st.session_state.rag_chain.invoke(question)
        st.write("### Response")
        st.write(response)

    # sidebar
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