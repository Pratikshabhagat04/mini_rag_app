import os
import shutil
from typing import Dict, List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)

STRICT_RAG_PROMPT = """You are a strict grounded AI assistant. Answer the user question ONLY based on the context provided below.

Rules you MUST follow:
1. If the answer cannot be found or inferred directly from the provided context, respond EXACTLY with:
   "I cannot answer this based on the provided documents."
2. Do NOT use outside knowledge or speculate.
3. Keep the answer concise, direct, and factual.

Context:
{context}

Question:
{question}

Answer:"""


def response_content_to_text(content: Any) -> str:
    """Return text from either legacy string or Gemini structured content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)

    return str(content)


class RAGPipeline:
    def __init__(self, persist_directory: str = VECTOR_STORE_DIR):
        self.persist_directory = persist_directory
        
        # Local embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        # Gemini LLM initialization
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )
        
        self.prompt = PromptTemplate.from_template(STRICT_RAG_PROMPT)
        self.vector_store = None
        self._load_existing_store()

    def _load_existing_store(self):
        """Loads vector store if it already exists on disk."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    def ingest_document(self, file_path: str) -> int:
        """Clears old persistent vector store and creates a fresh collection for the new document."""
        # 1. Clear previous collection from memory and disk
        if self.vector_store is not None:
            try:
                self.vector_store.delete_collection()
            except Exception:
                pass
            self.vector_store = None

        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        # 2. Load new document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .pdf or .txt file.")

        documents = loader.load()
        
        # 3. Split document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        # 4. Save to Chroma DB
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return len(chunks)

    def query(self, question: str, k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """Queries vector store using similarity search."""
        if not self.vector_store:
            return {
                "answer": "No documents ingested yet. Please upload a document first.",
                "sources": []
            }

        # Perform standard similarity search to retrieve top-k chunks
        docs = self.vector_store.similarity_search(question, k=k)
        
        context_blocks = []
        sources = []
        
        for doc in docs:
            context_blocks.append(doc.page_content)
            sources.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Uploaded File"),
                "page": doc.metadata.get("page", 0) + 1
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)
        formatted_prompt = self.prompt.format(context=formatted_context, question=question)
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response_content_to_text(response.content).strip(),
            "sources": sources
        }
