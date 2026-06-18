# PDF Helpdesk RAG Chatbot

A local Python-based Retrieval-Augmented Generation chatbot that answers questions from a PDF knowledge base. The project demonstrates a basic RAG workflow using PDF loading, text chunking, embeddings, vector storage, retrieval, chat history handling, and response generation with Gemini and LangChain.

This project was created as part of my Generative AI learning journey to understand how document question-answering systems work internally.

---

## Features

* Load a PDF document as a knowledge base
* Split PDF content into smaller overlapping chunks
* Generate embeddings for document chunks
* Store embeddings in a local Chroma vector database
* Retrieve relevant chunks based on user questions
* Use chat history to handle follow-up questions
* Generate answers using retrieved PDF context
* Local desktop-style GUI using Python
* Option to rebuild the vector database when needed

---

## Tech Stack

* Python
* LangChain
* Gemini API
* ChromaDB
* PyPDFLoader
* RecursiveCharacterTextSplitter
* Tkinter
* dotenv

---

## Project Pipeline

```text
PDF document
    |
Load PDF pages
    |
Split text into chunks
    |
Generate embeddings
    |
Store embeddings in Chroma vector database
    |
User asks a question
    |
Convert follow-up question into standalone question
    |
Retrieve relevant chunks from vector database
    |
Send question + retrieved context to LLM
    |
Generate answer based on PDF content
    |
Store question and answer in chat history
```

---

## Project Structure

```text
pdf_helpdesk_rag_chatbot/
│
├── data/
│   └── rag_practice_knowledge_base.pdf
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/sriyankabaral/pdf_helpdesk_rag_chatbot.git
cd pdf_helpdesk_rag_chatbot
```

### 2. Create a virtual environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a `.env` file in the project root and add:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=your_gemini_model_here
```

Do not upload your real API key to GitHub.

### 5. Run the app

```bash
python app.py
```

To rebuild the vector database:

```bash
python app.py --rebuild
```

---

## How It Works

The chatbot first loads the PDF document and splits it into chunks. Each chunk is converted into an embedding and stored in a Chroma vector database. When the user asks a question, the retriever searches for the most relevant chunks. The language model then uses only the retrieved context to generate an answer.

The project also includes a history-aware retriever, which helps convert follow-up questions into standalone questions before retrieval.

---

## Example Use Cases

This type of chatbot can be useful for:

* Company policy document question answering
* Student handbook chatbot
* Course material assistant
* FAQ chatbot
* Internal helpdesk document assistant
* Research paper or report question answering

---

## What I Learned

Through this project, I practiced:

* Building a basic RAG pipeline
* Loading and processing PDF documents
* Creating embeddings
* Using vector databases for semantic search
* Connecting retrieval with an LLM
* Handling chat history in a simple chatbot
* Building a local Python GUI application

---

## Limitations

* The chatbot currently works with one PDF file at a time
* The interface is local and not deployed online
* Chat history is stored only during the current app session
* The project is mainly for learning and portfolio demonstration
* More testing is needed for large documents and multiple file uploads

---

## Future Improvements

* Add support for multiple PDF uploads
* Add Streamlit or FastAPI version
* Store chat history persistently
* Show retrieved source chunks in the answer
* Add better error handling
* Add unit tests
* Improve project structure by separating UI, RAG pipeline, and configuration files

---

## Author

Sriyanka Baral
Recent BE graduate in Electronics, Communication and Information Technology
Interested in AI/ML, Computer Vision, Medical Image Analysis, and Generative AI applications
