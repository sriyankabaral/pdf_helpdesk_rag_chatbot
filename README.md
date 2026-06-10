# PDF Helpdesk RAG Chatbot

A local desktop-style Python chatbot that answers questions from `data/rag_practice_knowledge_base.pdf` using a Retrieval-Augmented Generation pipeline with the free Gemini API.

Run `app.py` and a window opens on your computer. This is not deployed or hosted.

## Project Structure

```text
pdf-helpdesk-rag-chatbot/
|-- data/
|   `-- rag_practice_knowledge_base.pdf
|-- app.py
|-- requirements.txt
|-- .env
|-- .gitignore
`-- README.md
```

## Pipeline

```text
PDF file
 |
Load PDF
 |
Split PDF into chunks
 |
Create embeddings for each chunk
 |
Store chunks in vector database
 |
User asks question
 |
Use chat history + question to create standalone question
 |
Retriever searches vector database
 |
Retriever returns top relevant chunks
 |
LLM reads question + chunks
 |
LLM gives final answer
 |
Save question and answer in chat history
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your Gemini API key and model to `.env`.

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

4. Open the local interface.

```bash
python app.py
```

The window includes these tabs:

- Conversation: ask questions and see answers.
- Database: view the PDF path and rebuild the vector database.
- Chat History: view or clear the current chat session.
- Configure: view the active Gemini configuration.

If you change embedding models or need to recreate the vector database, run:

```bash
python app.py --rebuild
```

## How It Works

- `PyPDFLoader` loads the PDF pages.
- `RecursiveCharacterTextSplitter` splits the PDF into overlapping text chunks.
- `GoogleGenerativeAIEmbeddings` creates Gemini embeddings for the chunks using `models/gemini-embedding-001`.
- `Chroma` stores and searches the embedded chunks.
- A history-aware retriever rewrites follow-up questions into standalone questions.
- `ChatGoogleGenerativeAI` answers using the retrieved PDF context with `gemini-3.5-flash` by default.
- A Python list stores question and answer chat history during the GUI session.
