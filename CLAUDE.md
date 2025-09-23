# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) System - a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. The system uses ChromaDB for vector storage, Anthropic's Claude AI for generation, and provides a web interface for interaction.

## Commands

### Development Commands
```bash
# Install dependencies
uv sync

# Start the application (recommended)
chmod +x run.sh
./run.sh

# Start manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code Quality Commands
```bash
# Format code (Black + isort)
./scripts/format.sh

# Run linting checks (flake8 + mypy)
./scripts/lint.sh

# Run all quality checks (format check + lint + tests) - strict mode
./scripts/quality.sh

# Comprehensive quality assessment - reports status without failing
./scripts/quality-check.sh

# Individual tool commands
uv run black .                    # Format with Black
uv run isort .                    # Sort imports
uv run flake8 backend/ main.py    # Lint with flake8
uv run mypy backend/ main.py      # Type check with mypy
uv run pytest -v                  # Run tests
```

### Environment Setup
- Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
- Application runs on `http://localhost:8000`
- API docs available at `http://localhost:8000/docs`

## Architecture

### Core System Components

The RAG system follows a modular architecture with these key components:

1. **RAGSystem** (`backend/rag_system.py`) - Main orchestrator that coordinates all components
2. **VectorStore** (`backend/vector_store.py`) - ChromaDB wrapper for vector storage and similarity search
3. **DocumentProcessor** (`backend/document_processor.py`) - Handles document parsing and chunking
4. **AIGenerator** (`backend/ai_generator.py`) - Anthropic Claude API integration with tool support
5. **SessionManager** (`backend/session_manager.py`) - Manages conversation history across sessions
6. **ToolManager & CourseSearchTool** (`backend/search_tools.py`) - Tool-based search implementation

### Data Flow

1. Documents in `/docs` are automatically loaded on startup
2. DocumentProcessor chunks documents and creates Course/CourseChunk objects
3. VectorStore stores both course metadata and content chunks in separate ChromaDB collections
4. User queries are processed by RAGSystem using AI tools for search
5. CourseSearchTool performs semantic search and returns relevant sources
6. AIGenerator uses Claude with tool definitions to provide contextual responses

### Key Models

- **Course** (`backend/models.py`) - Represents a complete course with metadata
- **CourseChunk** (`backend/models.py`) - Represents a chunk of course content with position tracking
- **SearchResults** (`backend/vector_store.py`) - Container for search results with metadata

### Configuration

All settings are centralized in `backend/config.py`:
- Chunk size: 800 characters with 100 character overlap
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Claude model: `claude-sonnet-4-20250514`
- Max conversation history: 2 messages
- ChromaDB path: `./chroma_db`

### Frontend

Simple HTML/CSS/JavaScript interface in `/frontend` directory served as static files by FastAPI.

### API Endpoints

- `POST /api/query` - Process user queries with session support
- `GET /api/courses` - Get course analytics and statistics

## Query Processing Flow

### Complete User Query Journey

1. **Frontend Input** (`frontend/script.js:45`) - User types query, triggers `sendMessage()`
2. **API Request** - POST to `/api/query` with `{query, session_id}`
3. **FastAPI Endpoint** (`backend/app.py:56`) - Creates session if needed, calls RAGSystem
4. **RAG Orchestration** (`backend/rag_system.py:102`) - Gets conversation history, formats prompt
5. **AI Generation** (`backend/ai_generator.py:43`) - Claude API call with tool definitions
6. **Tool Decision** - Claude autonomously decides whether to search based on query content
7. **Search Execution** (if needed) - `CourseSearchTool` performs semantic search in ChromaDB
8. **Response Synthesis** - Second Claude API call with search results to generate final answer
9. **Source Tracking** - Sources collected and returned to frontend for display
10. **Session Update** - Conversation history updated for context in future queries

### Document Processing Pipeline

Documents follow this structured format and processing:

**Expected Document Structure:**
```
Course Title: [title]
Course Link: [url] (optional)
Course Instructor: [name] (optional)

Lesson 0: Introduction
Lesson Link: [url] (optional)
[lesson content...]

Lesson 1: Next Topic
[lesson content...]
```

**Processing Steps:**
1. **File Reading** (`backend/document_processor.py:13`) - UTF-8 with fallback error handling
2. **Metadata Extraction** - Parses course title, link, instructor from first 3-4 lines
3. **Lesson Parsing** - Identifies `Lesson X:` markers and optional lesson links
4. **Sentence-Based Chunking** - 800 chars with 100 char overlap, preserves sentence boundaries
5. **Context Enhancement** - Adds prefixes like `"Course [title] Lesson X content: [chunk]"`
6. **Object Creation** - Course, Lesson, and CourseChunk objects with metadata

### Tool-Based Search Architecture

The system uses **intelligent tool selection** rather than always searching:

- **CourseSearchTool** (`backend/search_tools.py:20`) - Semantic search with course/lesson filtering
- **ToolManager** (`backend/search_tools.py:116`) - Registers and executes tools
- **Decision Logic** - Claude decides when to search vs. answer from general knowledge
- **One Search Rule** - Maximum one search per query to prevent tool loops
- **Source Tracking** - `last_sources` tracked and reset after each response

### ChromaDB Vector Storage

- **Dual Collections** - Separate storage for course metadata and content chunks
- **Embedding Model** - `all-MiniLM-L6-v2` sentence transformers
- **Search Results** - `SearchResults` class with documents, metadata, distances
- **Deduplication** - Prevents re-processing existing courses by title

### Session Management

- **Conversation Context** - Up to 2 previous message exchanges maintained
- **Session Creation** - Auto-generated session IDs (`session_1`, `session_2`, etc.)
- **History Format** - `"User: question\nAssistant: answer"` format for Claude context

## Important Implementation Details

- The system uses tool-based search where Claude decides when to search the knowledge base
- ChromaDB collections separate course metadata from content chunks for efficient retrieval
- Session management enables contextual conversations
- Documents are automatically deduplicated based on course titles
- The application automatically loads documents from `/docs` on startup
- Smart chunking preserves sentence boundaries and adds contextual prefixes
- Tool execution limited to one search per query for performance and loop prevention
- make sure uv to manage all dependencies