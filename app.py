import argparse
import os
import queue
import shutil
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

APP_DIR = Path(__file__).parent
PDF_PATH = APP_DIR / "data" / "rag_practice_knowledge_base.pdf"
CHROMA_PATH = APP_DIR / "chroma_db"
COLLECTION_NAME = "rag_practice_knowledge_base"
DEFAULT_CHAT_MODEL = "gemini-3.5-flash"
MAX_EMBEDDING_RETRIES = 4


class RetryingEmbeddings(Embeddings):
    def __init__(self, embeddings, max_retries=MAX_EMBEDDING_RETRIES):
        self.embeddings = embeddings
        self.max_retries = max_retries

    def _run_with_retries(self, operation):
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except Exception:
                if attempt == self.max_retries:
                    raise
                wait_seconds = 2 ** attempt
                print(
                    f"Gemini embedding request failed. Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

    def embed_documents(self, texts):
        return self._run_with_retries(lambda: self.embeddings.embed_documents(texts))

    def embed_query(self, text):
        return self._run_with_retries(lambda: self.embeddings.embed_query(text))


def load_pdf_chunks():
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(pages)


def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return RetryingEmbeddings(embeddings)


def get_llm():
    model = os.getenv("GEMINI_MODEL", DEFAULT_CHAT_MODEL).strip()
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def has_google_api_key():
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    return bool(api_key and api_key != "your_google_api_key_here")


def get_vector_store(embeddings, rebuild=False):
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing PDF: {PDF_PATH}")

    if rebuild and CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    vector_database_exists = CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())
    if vector_database_exists:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings,
        )
        try:
            if vector_store._collection.count() > 0:
                return vector_store
        except Exception:
            pass

        shutil.rmtree(CHROMA_PATH)

    chunks = load_pdf_chunks()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
    )


def build_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, rewrite the "
                "latest question as a standalone question. Do not answer it.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful PDF helpdesk assistant. Answer using only the "
                "retrieved context. If the answer is not in the context, say you "
                "do not know based on the PDF.\n\nContext:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)


def ask_question(rag_chain, question, chat_history):
    response = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history,
        }
    )
    answer = response["answer"]
    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
    )
    return answer


class RAGDesktopApp:
    def __init__(self, root, rebuild=False):
        self.root = root
        self.rebuild = rebuild
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.rag_chain = None
        self.chat_history = []
        self.transcript = []
        self.result_queue = queue.Queue()

        self.root.title("ChatWithYourData_Bot")
        self.root.geometry("920x620")
        self.root.minsize(760, 500)

        self.build_ui()
        self.refresh_database_status()
        self.poll_result_queue()

    def build_ui(self):
        header = ttk.Label(
            self.root,
            text="ChatWithYourData_Bot",
            font=("Segoe UI", 18, "bold"),
        )
        header.pack(anchor="w", padx=14, pady=(14, 8))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.conversation_tab = ttk.Frame(self.notebook, padding=10)
        self.database_tab = ttk.Frame(self.notebook, padding=10)
        self.history_tab = ttk.Frame(self.notebook, padding=10)
        self.configure_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.conversation_tab, text="Conversation")
        self.notebook.add(self.database_tab, text="Database")
        self.notebook.add(self.history_tab, text="Chat History")
        self.notebook.add(self.configure_tab, text="Configure")

        self.build_conversation_tab()
        self.build_database_tab()
        self.build_history_tab()
        self.build_configure_tab()

    def build_conversation_tab(self):
        input_frame = ttk.Frame(self.conversation_tab)
        input_frame.pack(fill="x")

        self.question_entry = ttk.Entry(input_frame)
        self.question_entry.pack(side="left", fill="x", expand=True)
        self.question_entry.insert(0, "Enter text here...")
        self.question_entry.bind("<FocusIn>", self.clear_placeholder)
        self.question_entry.bind("<Return>", lambda _event: self.submit_question())

        self.ask_button = ttk.Button(input_frame, text="Ask", command=self.submit_question)
        self.ask_button.pack(side="left", padx=(8, 0))

        separator = ttk.Separator(self.conversation_tab)
        separator.pack(fill="x", pady=12)

        text_frame = ttk.Frame(self.conversation_tab)
        text_frame.pack(fill="both", expand=True)

        self.conversation_text = tk.Text(
            text_frame,
            wrap="word",
            height=18,
            state="disabled",
            font=("Segoe UI", 10),
        )
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.conversation_text.yview
        )
        self.conversation_text.configure(yscrollcommand=scrollbar.set)

        self.conversation_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def build_database_tab(self):
        self.database_status = tk.StringVar()
        ttk.Label(
            self.database_tab,
            text="Knowledge Base",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(self.database_tab, text=str(PDF_PATH)).pack(anchor="w", pady=(4, 12))
        ttk.Label(self.database_tab, textvariable=self.database_status).pack(anchor="w")
        ttk.Button(
            self.database_tab,
            text="Rebuild Vector Database",
            command=self.rebuild_database,
        ).pack(anchor="w", pady=(12, 0))

    def build_history_tab(self):
        self.history_text = tk.Text(
            self.history_tab,
            wrap="word",
            height=20,
            state="disabled",
            font=("Segoe UI", 10),
        )
        self.history_text.pack(fill="both", expand=True)
        ttk.Button(
            self.history_tab,
            text="Clear Chat History",
            command=self.clear_chat_history,
        ).pack(anchor="w", pady=(10, 0))

    def build_configure_tab(self):
        api_status = "Set" if has_google_api_key() else "Missing"
        model = os.getenv("GEMINI_MODEL", DEFAULT_CHAT_MODEL).strip()

        ttk.Label(
            self.configure_tab,
            text="Gemini Configuration",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(self.configure_tab, text=f"GOOGLE_API_KEY: {api_status}").pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(self.configure_tab, text=f"GEMINI_MODEL: {model}").pack(anchor="w")
        ttk.Label(
            self.configure_tab,
            text="Edit .env and restart this program to change these values.",
        ).pack(anchor="w", pady=(12, 0))

    def clear_placeholder(self, _event):
        if self.question_entry.get() == "Enter text here...":
            self.question_entry.delete(0, tk.END)

    def refresh_database_status(self):
        if not PDF_PATH.exists():
            status = "PDF missing"
        elif CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
            status = "Vector database found"
        else:
            status = "Vector database will be built on first question"
        self.database_status.set(status)

    def append_conversation(self, speaker, message):
        self.conversation_text.configure(state="normal")
        self.conversation_text.insert(tk.END, f"{speaker}:\t{message}\n\n")
        self.conversation_text.see(tk.END)
        self.conversation_text.configure(state="disabled")

    def refresh_history_tab(self):
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", tk.END)
        self.history_text.insert(tk.END, "\n\n".join(self.transcript))
        self.history_text.configure(state="disabled")

    def set_busy(self, is_busy):
        state = "disabled" if is_busy else "normal"
        self.ask_button.configure(state=state)
        self.question_entry.configure(state=state)

    def ensure_rag_chain(self, rebuild=False):
        if rebuild:
            self.embeddings = None
            self.llm = None
            self.vector_store = None
            self.rag_chain = None

        if self.rag_chain is None:
            self.embeddings = get_embeddings()
            self.llm = get_llm()
            self.vector_store = get_vector_store(self.embeddings, rebuild=rebuild)
            self.rag_chain = build_rag_chain(self.vector_store, self.llm)
            self.rebuild = False
        return self.rag_chain

    def submit_question(self):
        question = self.question_entry.get().strip()
        if not question or question == "Enter text here...":
            return

        if not has_google_api_key():
            messagebox.showerror(
                "Missing API Key",
                "Add GOOGLE_API_KEY to .env, then restart this program.",
            )
            return

        self.question_entry.delete(0, tk.END)
        self.append_conversation("User", question)
        self.transcript.append(f"User: {question}")
        self.refresh_history_tab()
        self.set_busy(True)

        worker = threading.Thread(
            target=self.answer_question_worker,
            args=(question, self.rebuild),
            daemon=True,
        )
        worker.start()

    def answer_question_worker(self, question, rebuild):
        try:
            rag_chain = self.ensure_rag_chain(rebuild=rebuild)
            answer = ask_question(rag_chain, question, self.chat_history)
            self.result_queue.put(("answer", answer))
        except Exception as error:
            self.result_queue.put(("error", str(error)))

    def poll_result_queue(self):
        try:
            while True:
                event_type, payload = self.result_queue.get_nowait()
                if event_type == "answer":
                    self.append_conversation("ChatBot", payload)
                    self.transcript.append(f"ChatBot: {payload}")
                    self.refresh_history_tab()
                    self.refresh_database_status()
                else:
                    self.append_conversation("Error", payload)
                    messagebox.showerror("Chatbot Error", payload)
                self.set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_result_queue)

    def rebuild_database(self):
        self.rebuild = True
        self.chat_history.clear()
        self.transcript.clear()
        self.refresh_history_tab()
        self.conversation_text.configure(state="normal")
        self.conversation_text.delete("1.0", tk.END)
        self.conversation_text.configure(state="disabled")
        self.refresh_database_status()
        messagebox.showinfo(
            "Rebuild Queued",
            "The vector database will be rebuilt on your next question.",
        )

    def clear_chat_history(self):
        self.chat_history.clear()
        self.transcript.clear()
        self.refresh_history_tab()


def parse_args():
    parser = argparse.ArgumentParser(description="Open a local PDF RAG chatbot GUI.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the local Chroma vector database on first question.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    RAGDesktopApp(root, rebuild=args.rebuild)
    root.mainloop()


if __name__ == "__main__":
    main()
