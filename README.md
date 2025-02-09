
# **Document-Based Q&A System Using Langchain, Mistral AI, and Streamlit**  

This project provides a complete solution for building a question-answering (Q&A) system that extracts relevant information from PDF documents and answers user queries. It combines **Langchain** for document processing, **Mistral AI** for generating responses, **FAISS** for fast search, **Hugging Face** for embeddings, and **Streamlit** for an intuitive user interface.

🔗 **View the live demo:** [https://shadesofprakash.streamlit.app/](https://shadesofprakash.streamlit.app/)  

---

## **Features**  
- **PDF Upload and Processing**: Upload PDF documents and split them into manageable chunks for efficient processing.  
- **FAISS for Fast Information Retrieval**: Build and query an index for fast document search and retrieval.  
- **Mistral AI for Intelligent Responses**: Generate accurate, context-aware answers based on the document content.  
- **Streamlit UI**: A user-friendly web interface for real-time question-answering.

---

## **Project Structure**  
```
📁 project-root/
  ├── unit-1.pdf            # Sample PDF file for document processing
  ├── main.py               # Main script to run the Q&A system
  ├── .env                  # Environment file to store sensitive variables
  └── requirements.txt      # Python dependencies
```

---

## **Setup Instructions**  

### **1. Prerequisites**  
- Python 3.9 or higher  
- pip (Python package manager)  

### **2. Clone the Repository**  
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### **3. Install Required Packages**  
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

### **1. Run the Streamlit Application**  
Start the application using the following command:  
```bash
streamlit run main.py
```

The web app allows you to upload a PDF, enter a question, and receive an answer in real time.

---

## **Example Interaction**  
1. Upload `unit-1.pdf` through the Streamlit UI.  
2. Enter a question like **"What is machine learning?"**  
3. The system retrieves the relevant context and generates an answer using Mistral AI.

---

## **Dependencies**  
- `langchain_community`  
- `langchain_core`  
- `langchain_mistralai`  
- `sentence-transformers`  
- `faiss-cpu`  
- `streamlit`  
- `python-dotenv`  


