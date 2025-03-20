import pytest
from pathlib import Path
from src.core.rag_agent import RAGAgent

def test_rag_agent_initialization():
    agent = RAGAgent()
    assert agent.llm is not None
    assert agent.embeddings is not None
    assert agent.text_splitter is not None
    assert agent.vectorstore is None

def test_directories_creation():
    agent = RAGAgent()
    assert Path(agent.persist_directory).exists()
    assert Path(agent.directory_path).exists()

def test_document_loading(tmp_path):
    # Create a test document
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")
    
    agent = RAGAgent()
    documents = agent.load_documents([str(test_file)])
    assert len(documents) == 1
    assert documents[0].page_content == "This is a test document." 