import os
import tempfile
import streamlit as st
from src.rag_engine import RAGPipeline

st.set_page_config(page_title="Mini RAG Application", page_icon="📚", layout="wide")

@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

rag = get_rag_pipeline()

st.title("📚 Mini Retrieval-Augmented Generation (RAG) System")
st.caption("Ask grounded questions about your documents with strict hallucination guardrails.")

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Document", type="primary"):
            with st.spinner("Chunking & Embedding Document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    num_chunks = rag.ingest_document(tmp_path)
                    st.success(f"Indexed **{uploaded_file.name}** into **{num_chunks} chunks**!")
                except Exception as e:
                    st.error(f"Failed to process document: {str(e)}")
                finally:
                    os.unlink(tmp_path)

    st.markdown("---")
    st.markdown("### Model Config")
    st.text("Embeddings: all-MiniLM-L6-v2")
    st.text("LLM: Gemini 3.8 Flash")

# --- Main Section: Search & QA ---
st.header("2. Query Knowledge Base")
user_query = st.text_input("Ask a question based on your indexed files:", placeholder="e.g., What is the candidate's work experience?")

if user_query:
    with st.spinner("Retrieving context & generating answer..."):
        result = rag.query(user_query)
        
        # Display Answer
        st.subheader("Answer:")
        if "cannot answer" in result["answer"].lower():
            st.warning(result["answer"])
        else:
            st.success(result["answer"])

        # Display Sources / Citations (Bonus Feature)
        if result["sources"]:
            with st.expander("🔍 View Retrieved Context & Citation Metadata"):
                for idx, src in enumerate(result["sources"]):
                    st.markdown(f"**Chunk {idx+1}** | Source: `{os.path.basename(src['source'])}` (Page {src['page']})")
                    st.info(src["content"])
