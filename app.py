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

# Load API key from environment
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Custom CSS for responsiveness
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }

        h1 {
            color: #4A90E2;
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        p {
            font-size: 18px;
            color: #c5cbd2;
        }

        .stTextInput, .stFileUploader {
            width: 100%;
        }      
        
        @media (max-width: 480px) {
            h1 {
                font-size: 20px;
               margin-bottom: 40px;
            }
            p {
                font-size: 12px;
            }
            .stTextInput, .stButton {
                font-size: 12px;
                padding: 10px;
            }
        }

        .main {
            padding: 15px;
        }

        .footer {
            text-align: center;
            color: #839ab2;
            margin-top: 200px;
        }
    </style>
""", unsafe_allow_html=True)

# Main app title and description
st.markdown("<h1 style='text-align: center;'>📝 Document Q&A Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your document and ask questions to get instant answers!</p>", unsafe_allow_html=True)

# File uploader for PDF or text files
uploaded_file = st.file_uploader("Upload your document (PDF or TXT)", type=['pdf', 'txt'], label_visibility='collapsed')

# Helper class for text documents
class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}  # Metadata placeholder

if uploaded_file:
    with st.spinner('Processing document...'):
        # Load document content based on file type
        if uploaded_file.type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        elif uploaded_file.type == 'text/plain':
            text = uploaded_file.read().decode("utf-8")
            documents = [Document(text)]  # Wrap text in a Document object

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)

        # Initialize the language model
        llm = ChatGroq(temperature=0.5, model_name="mixtral-8x7b-32768")

        # Prompt template for questions
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        # Set up the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success("Document processed! Now ask your questions.")

    # Input field for questions
    question = st.text_input("Ask a question about your document:")

    if question:
        start = time.process_time()
        with st.spinner('Generating answer...'):
            response = retrieval_chain.invoke({'input': question})
        st.write(f"**Answer:** {response['answer']}")
        st.write(f"_Response generated in {time.process_time() - start:.2f} seconds_")

        # Display document context in an expandable section
        with st.expander("View related document context"):
            for i, doc in enumerate(response['context']):
                st.write(f"**Context {i+1}:** {doc.page_content}")
                st.write("---")

# Footer
st.markdown("<hr><p class='footer'>Built with 💙 using LangChain & Groq</p>", unsafe_allow_html=True)
