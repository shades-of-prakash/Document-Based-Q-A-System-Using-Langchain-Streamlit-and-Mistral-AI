import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai import ChatMistralAI
from langchain.chains  import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('MISTRAL_API_KEY')

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found. Please set it in your .env file.")


# Load the PDF file and split it into chunks
loader = PyPDFLoader("sample.pdf")
documents = loader.load()


# Split the documents into smaller chunks (e.g., 500 characters per chunk)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)


# # Use Sentence Transformers for embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index_path = "faiss_index"

# Step 2: Check if the FAISS index already exists
db = FAISS.from_documents(chunks, embedding_model)
#save it locally
db.save_local(index_path)

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

llm=ChatMistralAI(model="mistral-large-latest",api_key=api_key)

document_chain=create_stuff_documents_chain(llm,prompt)

#retriever

retriever=db.as_retriever()
    
#retrieval_chain

retrieval_chain=create_retrieval_chain(retriever,document_chain)

while True:
    user_input = input("Enter your question (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    result = retrieval_chain.invoke({"input": user_input})
    print("\nAnswer:\n", result.get('answer', 'No answer found.\n'))