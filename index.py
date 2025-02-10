import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get('MISTRAL_API_KEY')

# Streamlit page configuration
st.set_page_config(page_title="AI PDF Assistant", layout="centered")
st.title("📄 AI PDF Assistant with Mistral AI")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Check for API key
if not api_key:
    st.error("MISTRAL_API_KEY not found. Please set it in your .env file.")
    st.stop()

def process_pdf(uploaded_file):
    """Process the uploaded PDF and create the retrieval chain."""
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever()

        prompt =ChatPromptTemplate.from_template("""
            You are an intelligent assistant tasked with providing detailed and comprehensive answers based **solely** on the provided context. Follow these guidelines to ensure your response is thorough, accurate, and well-structured:
            ### Instructions:
            1. **Focus exclusively on the given context. Avoid making assumptions or using external information.**
            2. **Think deeply and reason through the answer step by step.** Break down complex ideas into smaller, manageable parts.
            3. If the question cannot be answered from the context, respond with: "The context does not contain enough information to answer this question."
            4. **Structure your answer clearly and comprehensively.** Use detailed explanations, bullet points, examples, and any relevant data or statistics from the context.
            5. **Elaborate on key points** to provide a deeper understanding. Include background information, implications, and potential follow-up considerations.
            6. **Summarize the main points** at the end of your response to reinforce the key takeaways.
            ---
            ### Provided Context:
            {context}
            ---
            ### Question:
            {input}
            ---
            ### Answer:
            # """)


        llm = ChatMistralAI(model="mistral-large-latest", api_key=api_key)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        return create_retrieval_chain(retriever, document_chain)

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF..."):
        st.session_state.retrieval_chain = process_pdf(uploaded_file)
    if st.session_state.retrieval_chain:
        st.session_state.pdf_processed = True
        st.success("PDF processed successfully! You can now ask questions about it.")
        os.remove("temp.pdf")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.retrieval_chain:
    if prompt := st.chat_input("Ask a question about your PDF"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.retrieval_chain.invoke({"input": prompt})
                answer = response.get('answer', 'I cannot answer this question based on the provided context.')
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
