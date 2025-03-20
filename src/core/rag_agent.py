import json
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from .config import MODEL_NAME, TEMPERATURE, EMBEDDING_MODEL, VECTOR_DB_DIR, DOCUMENTS_DIR

class RAGAgent:
    def __init__(self, model_name=MODEL_NAME, temperature=TEMPERATURE, 
                 embedding_model=EMBEDDING_MODEL, persist_directory=str(VECTOR_DB_DIR)):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.directory_path = str(DOCUMENTS_DIR)
        self._ensure_directories_exist()
        test_embedding = self.embeddings.embed_query("Test sentence")
        print(f"Embedding model test: Length={len(test_embedding)}, Sample={test_embedding[:5]}")

    def _ensure_directories_exist(self):
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.directory_path).mkdir(parents=True, exist_ok=True)

    def load_documents(self, file_paths=None):
        documents = []
        loaded_paths = set()
        if file_paths:
            for file_path in file_paths:
                path = Path(file_path)
                if not path.exists():
                    print(f"Warning: File {file_path} does not exist.")
                    continue
                if str(path) in loaded_paths:
                    continue
                if path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(path))
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded PDF {file_path}: {len(docs)} pages")
                elif path.suffix.lower() == '.txt':
                    loader = TextLoader(str(path))
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded TXT {file_path}: {docs[0].page_content[:100]}...")
                loaded_paths.add(str(path))
        
        dir_path = Path(self.directory_path)
        if dir_path.exists() and dir_path.is_dir():
            pdf_loader = DirectoryLoader(str(dir_path), glob="**/*.pdf", loader_cls=PyPDFLoader)
            txt_loader = DirectoryLoader(str(dir_path), glob="**/*.txt", loader_cls=TextLoader)
            for doc in pdf_loader.load() + txt_loader.load():
                if doc.metadata['source'] not in loaded_paths:
                    documents.append(doc)
                    loaded_paths.add(doc.metadata['source'])
                    print(f"Loaded from ./documents/: {doc.metadata['source']} - {doc.page_content[:100]}...")
        print(f"Total documents loaded: {len(documents)}")
        return documents

    def _get_file_metadata(self, file_paths):
        metadata = {}
        if file_paths:
            for fp in file_paths:
                path = Path(fp)
                if path.exists():
                    metadata[str(path)] = path.stat().st_mtime
        dir_path = Path(self.directory_path)
        if dir_path.exists() and dir_path.is_dir():
            for ext in ['*.pdf', '*.txt']:
                for file in dir_path.rglob(ext):
                    metadata[str(file)] = file.stat().st_mtime
        return metadata

    def _load_previous_metadata(self):
        metadata_file = Path(self.persist_directory) / "processed_files.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata):
        metadata_file = Path(self.persist_directory) / "processed_files.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

    def process_documents(self, documents):
        if not documents:
            print("No documents to process")
            return
        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            print("No chunks created from documents")
            return
        
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk.page_content[:100]}... (length: {len(chunk.page_content)})")
            embedding = self.embeddings.embed_query(chunk.page_content)
            print(f"Embedding for chunk {i}: Length={len(embedding)}, Sample={embedding[:5]}")
        
        if Path(self.persist_directory).exists():
            if not self.vectorstore:
                self.load_existing_vectorstore()
            self.vectorstore.add_documents(chunks)
            print(f"Added {len(chunks)} new chunks to existing vector store")
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Created new vector store with {len(chunks)} chunks")

    def update_documents(self, file_paths=None):
        current_metadata = self._get_file_metadata(file_paths)
        previous_metadata = self._load_previous_metadata()
        
        new_or_modified = {path: mtime for path, mtime in current_metadata.items()
                          if path not in previous_metadata or mtime > previous_metadata[path]}
        
        if new_or_modified:
            new_docs = self.load_documents(file_paths=list(new_or_modified.keys()))
            if not new_docs:
                return "No new documents loaded to process."
            self.process_documents(new_docs)
            self._save_metadata(current_metadata)
            return f"Processed {len(new_or_modified)} new or modified documents."
        return "No new or modified documents found."

    def load_existing_vectorstore(self):
        if Path(self.persist_directory).exists():
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Loaded existing vector store from {self.persist_directory}")
        else:
            raise ValueError("No vector store found. Please update documents first.")

    def get_retriever(self, search_k=4):
        if not self.vectorstore:
            self.load_existing_vectorstore()
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": search_k})

    def get_qa_chain(self, chat_history):
        retriever = self.get_retriever()
        prompt_template = """
        You are an intelligent assistant designed to answer user questions. Provide accurate and concise answers to the best of your ability. Use any relevant documents or your own general knowledge to answer the user’s question. You should not mention where the information comes from—just provide a seamless and natural response.
        •	Answer questions clearly and directly.
        •	Take the previous conversation into account for continuity and context.
        •	If you don’t have enough information to provide an answer, simply respond with: ‘I don’t have enough information at this time.’

        Context:

        {context}

        Previous Conversation:

        {chat_history}

        Based on the context, previous conversation, or your own general knowledge, answer the user’s question in a natural and engaging way.
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, PROMPT)
        return create_retrieval_chain(retriever, question_answer_chain) 