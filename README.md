# Code Optimizer

AI-powered code optimization tool using LangGraph, Gemini/Ollama, and Docker sandboxing.

## Features

- Runtime profiling with Docker sandbox
- LangGraph workflow for optimization pipeline
- FAISS vector store for pattern retrieval
- Gemini AI analysis with Ollama fallback
- Automated code optimization reports
- Dark-themed React frontend with Monaco editor

## Tech Stack

- **Backend:** Flask, LangGraph, FAISS
- **Frontend:** React, Vite, Monaco Editor
- **LLM:** Gemini (primary), Ollama (fallback)
- **Sandbox:** Docker for isolated code execution
- **Retrieval:** FAISS vector store with sentence-transformers

## Deployment

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 18+

### Quick Start

1. **Clone and setup environment:**
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

2. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

3. **Access the application:**
- Frontend: http://localhost:5173 (dev) or serve built files
- Backend API: http://localhost:5001
- Ollama: http://localhost:11434

### Manual Deployment

**Backend:**
```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_key"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5-coder:3b"

# Run Flask server
python -m api.app
```

**Frontend:**
```bash
cd frontend
npm install
npm run build
# Serve dist/ with nginx or any web server
```

**Docker Sandbox:**
```bash
cd sandbox
docker build -f Dockerfile.sandbox -t code-optimizer-sandbox .
```

**Ollama:**
```bash
# Install Ollama from https://ollama.com
ollama serve
ollama pull qwen2.5-coder:3b
```

### Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - Ollama model (default: qwen2.5-coder:3b)
- `SANDBOX_IMAGE` - Docker sandbox image (default: code-optimizer-sandbox)

## Project Structure

```
codeoptimiser/
├── agent/          # LangGraph pipeline
├── api/            # Flask API
├── llm/            # Gemini/Ollama clients
├── sandbox/        # Docker runner
├── retrieval/      # FAISS vector store
├── frontend/       # React app
└── tests/          # Test suite
```

## Testing

```bash
# Backend tests
pytest tests/

# Docker runner test
python tests/test_docker_runner.py

# Gemini API test
python tests/test_gemini_key
```