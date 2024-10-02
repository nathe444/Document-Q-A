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
import docx

# Load API key from environment
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Custom CSS for better styling
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #F7F9FC;
        }

        h1 {
            color: #4A90E2;
            font-size: 38px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }

        h2 {
            font-size: 26px;
            color: #333333;
            margin-bottom: 10px;
            text-align: center;
        }

        p {
            color: #5A6C7D;
            font-size: 18px;
            line-height: 1.6;
        }

        .stTextInput, .stFileUploader, .stButton {
            width: 100%;
            max-width: 700px;
            margin: 0 auto;
        }
            

        .stFileUploader {
            margin-top: 10px;
            padding: 10px;
        }

        .stButton {
            background-color: #4A90E2 !important;
            color: #ffffff !important;
            border-radius: 8px;
            padding: 12px !important;
            font-size: 16px;
            margin-top: 20px;
        }

        @media (max-width: 768px) {
            h1 {
                font-size: 28px;
            }

            h2 {
                font-size: 20px;
            }

            p {
                font-size: 16px;
            }

            .stTextInput, .stButton {
                padding: 10px;
                font-size: 14px;
            }
        }
        
        .footer {
            text-align: center;
            color: #839ab2;
            margin-top: 40px;
            font-size: 16px;
        }

        hr {
            margin-top: 30px;
            border: none;
            border-top: 1px solid #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1>📝 Document Q&A Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your document (PDF, DOCX, or TXT) and ask questions to get instant answers!</p>", unsafe_allow_html=True)

# File uploader section
uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'], label_visibility='collapsed')

# Document handler class
class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    extracted_text = '\n'.join([para.text for para in doc.paragraphs])
    return extracted_text

documents = []
if uploaded_file:
    with st.spinner('Processing document...'):
        if uploaded_file.type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':  # DOCX type
            extracted_text = extract_text_from_docx(uploaded_file)
            documents = [Document(extracted_text)]
        elif uploaded_file.type == 'text/plain':
            text = uploaded_file.read().decode("utf-8")
            documents = [Document(text)]

    if documents:
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

# Question input section (Always visible)
question = st.text_input("Ask a question about your document:")

# Answer generation
if question:
    if documents:
        start = time.process_time()
        with st.spinner('Generating answer...'):
            response = retrieval_chain.invoke({'input': question})
        st.write(f"**Answer:** {response['answer']}")
        st.write(f"_Response generated in {time.process_time() - start:.2f} seconds_")

        # Display related document context
        with st.expander("View related document context"):
            for i, doc in enumerate(response['context']):
                st.write(f"**Context {i+1}:** {doc.page_content}")
                st.write("---")
    else:
        st.write("**Note:** Please upload a document to ask document-related questions.")

# Footer
st.markdown("<hr><p class='footer'>Built with 💙 using LangChain & Groq</p>", unsafe_allow_html=True)
