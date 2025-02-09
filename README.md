# **Document-Based Q&A System Using Langchain and Mistral AI**

This repository provides a complete end-to-end solution for building a question-answering (Q&A) system using the following tools and technologies:

- **Langchain** for document loading, chunking, and chain creation  
- **Mistral AI** for generating intelligent responses  
- **FAISS (Facebook AI Similarity Search)** for efficient document retrieval  
- **Hugging Face Sentence Transformers** for embedding  
- **Python** for scripting  
- **dotenv** for environment variable management  

---

## **Features**

- **PDF Document Loading and Processing**  
  Load PDF documents and split them into manageable chunks for processing.
- **FAISS Index for Fast Retrieval**  
  Use FAISS to build and query an index for efficient information retrieval.
- **Mistral AI for Natural Language Understanding**  
  Generate context-aware, step-by-step answers using Mistral AI.
- **Real-time User Interaction**  
  Interactive command-line interface where users can ask questions based on the content of the loaded PDF.

---

## **Project Structure**

```bash
📁 project-root/
  ├── unit-1.pdf            # Sample PDF file for document processing
  ├── main.py               # Main script to run the Q&A system
  ├── .env                  # Environment file to store sensitive variables
  └── requirements.txt       # Python dependencies
```

---

## **Setup Instructions**

### **1. Prerequisites**
Make sure you have the following installed:

- Python 3.9 or higher  
- pip (Python package manager)

### **2. Clone the Repository**
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### **3. Install Required Packages**
You can generate a `requirements.txt` file using the following command if not already present:
```bash
pip freeze > requirements.txt
```

Or install the provided packages:
```bash
pip install -r requirements.txt
```

### **4. Add API Key to `.env` File**
Create a `.env` file in the project root directory and add your Mistral API key:

```
MISTRAL_API_KEY=your_api_key_here
```

---

## **Usage Instructions**

### **1. Load the PDF Document and Create a FAISS Index**

The script will load `unit-1.pdf`, split it into chunks of 500 characters with 50 characters overlapping, and create embeddings using **Hugging Face Sentence Transformers**. It then saves the FAISS index locally.

```python
loader = PyPDFLoader("unit-1.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

db = FAISS.from_documents(chunks, embedding_model)
db.save_local(index_path)
```

### **2. Start the Q&A System**

The system prompts the user to ask questions interactively:

```python
while True:
    user_input = input("Enter your question (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    result = retrieval_chain.invoke({"input": user_input})
    print("\nAnswer:\n", result.get('answer', 'No answer found.\n'))
```

### **3. Answer Generation**

The response generation is based on the provided context. If the context does not contain enough information, it will notify the user.

---

## **Prompt Template for Mistral AI**

The response generation is driven by the following prompt template:

```python
prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant tasked with answering questions based **only** on the provided context. Follow these rules to give a helpful, accurate, and well-structured response:
### Instructions:
1. **Focus only on the given context. Do not make assumptions.**
2. **Think step by step** and reason through the answer logically.
3. If the question is not answerable from the context, respond with: "The context does not contain enough information to answer this question."
4. **Structure your answer** clearly and concisely with relevant explanations, bullet points, or examples.
5. End the response with a **summary or key takeaway** if applicable.
---
### Provided Context:
{context}
---
### Question:
{input}
---
### Answer:
""")
```

---

## **Example Interaction**

**User Input:**  
```
What is machine learning?
```

**Response:**  
```
Machine learning is a field of artificial intelligence that focuses on creating systems capable of learning from data and improving their performance over time without being explicitly programmed. The context does not contain enough information to answer this question.
```

---

## **Error Handling**

- **Environment Variable Missing:**  
  If the `MISTRAL_API_KEY` is not set, the script will raise an error:
  ```python
  if not api_key:
      raise ValueError("MISTRAL_API_KEY not found. Please set it in your .env file.")
  ```

- **No FAISS Index Found:**  
  The script will create a new FAISS index if one does not exist:
  ```python
  if not os.path.exists(f"{index_path}.index"):
      print("No FAISS index found. Creating a new one...")
      db = FAISS.from_documents(chunks, embedding_model)
      db.save_local(index_path)
  ```

---

## **Dependencies**

- `langchain_community`  
- `langchain_core`  
- `langchain_mistralai`  
- `sentence-transformers`  
- `faiss-cpu`  
- `python-dotenv`  

---

## **License**

This project is licensed under the MIT License.

---

## **Contributing**

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## **Contact**

For questions or suggestions, feel free to reach out at:  
**Email:** your-email@example.com  
**GitHub:** [your-username](https://github.com/your-username)

