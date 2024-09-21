import time
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📝 Document Q&A Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your document and ask questions to get instant answers!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your document (PDF or TXT)", type=['pdf', 'txt'], label_visibility='collapsed')

# Simple document class to hold text content and metadata
class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}  # Add metadata attribute

if uploaded_file:
    with st.spinner('Processing document...'):
        if uploaded_file.type == 'application/pdf':
            # Save the uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        elif uploaded_file.type == 'text/plain':
            text = uploaded_file.read().decode("utf-8")
            # Create a single Document object with the text
            documents = [Document(text)]  # Wrap text in Document class

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)

        llm = ChatGroq(temperature=0.5, model_name="mixtral-8x7b-32768")

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success("Document processed! Now ask your questions.")

    question = st.text_input("Ask a question about your document:")

    if question:
        start = time.process_time()
        with st.spinner('Generating answer...'):
            response = retrieval_chain.invoke({'input': question})
        st.write(f"**Answer:** {response['answer']}")
        st.write(f"_Response generated in {time.process_time() - start:.2f} seconds_")

        with st.expander("View related document context"):
            for i, doc in enumerate(response['context']):
                st.write(f"**Context {i+1}:** {doc.page_content}")
                st.write("---")

st.markdown("<hr><p style='text-align: center;'>Built with 💙 using LangChain & Groq</p>", unsafe_allow_html=True)
