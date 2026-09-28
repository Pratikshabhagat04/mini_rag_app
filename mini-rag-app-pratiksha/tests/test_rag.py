import os
import pytest
from src.rag_engine import RAGPipeline

@pytest.fixture(scope="module")
def rag_instance():
    pipeline = RAGPipeline(persist_directory="./test_chroma_db")
    
    # Create temporary text file for testing
    sample_text = (
        "John Doe is a Senior Software Engineer with 8 years of experience in Python and AWS. "
        "He graduated from Stanford University in 2016 with a Bachelor's degree in Computer Science."
    )
    test_file = "test_doc.txt"
    with open(test_file, "w") as f:
        f.write(sample_text)
        
    pipeline.ingest_document(test_file)
    yield pipeline
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

def test_grounded_answer(rag_instance):
    res = rag_instance.query("Where did John Doe graduate from?")
    assert "Stanford" in res["answer"]

def test_hallucination_guardrail(rag_instance):
    res = rag_instance.query("What is John Doe's favorite food?")
    assert res["answer"] == "I cannot answer this based on the provided documents."