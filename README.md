# RAG Chat with Memory

A modern RAG (Retrieval-Augmented Generation) chat agent built with FastAPI, LangChain, and Ollama. This application provides a powerful interface for document-based question answering with conversation memory.

## Features

- Document ingestion and processing (PDF and TXT files)
- Vector storage using ChromaDB
- Conversation memory with configurable message limits
- RESTful API endpoints for chat and document management
- Support for multiple document formats
- Efficient document chunking and embedding
- Thread-based conversation management

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Hugging Face account (for embeddings)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/niq-rag-agent.git
cd niq-rag-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p documents vector_db
```

## Usage

1. Start the server with auto-reload for development:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

2. The API will be available at `http://localhost:8000`

3. API Documentation will be available at `http://localhost:8000/docs`

## API Endpoints

### POST /api/v1/chat
Send a chat message and receive a response based on the indexed documents.

Request body:
```json
{
    "question": "Your question here",
    "thread_id": "unique-thread-id"
}
```

### POST /api/v1/update-documents
Update or add new documents to the knowledge base.

Request body:
```json
{
    "file_paths": ["path/to/document1.pdf", "path/to/document2.txt"]
}
```

## Project Structure

```
niq-rag-agent/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── rag_agent.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── documents/
├── vector_db/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Configuration

The application can be configured through environment variables:

- `MODEL_NAME`: Ollama model name (default: "llama3.2")
- `TEMPERATURE`: Model temperature (default: 0.1)
- `EMBEDDING_MODEL`: Hugging Face embedding model (default: "all-MiniLM-L6-v2")
- `MAX_MESSAGES`: Maximum number of messages to keep in memory (default: 10)

## Development

The application runs in development mode with auto-reload enabled, which means:
- Any changes to the code will automatically restart the server
- The server watches the `src` directory for changes
- Detailed logs are displayed in the console

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 