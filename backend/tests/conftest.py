import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock()
    config.chunk_size = 800
    config.chunk_overlap = 100
    config.embedding_model = "all-MiniLM-L6-v2"
    config.claude_model = "claude-sonnet-4-20250514"
    config.max_conversation_history = 2
    config.chroma_db_path = "./test_chroma_db"
    config.anthropic_api_key = "test_api_key"
    return config


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for testing"""
    with patch('app.RAGSystem') as mock_rag_class:
        mock_rag_instance = Mock()
        mock_rag_class.return_value = mock_rag_instance

        # Mock session manager
        mock_session_manager = Mock()
        mock_rag_instance.session_manager = mock_session_manager

        # Default return values for common methods
        mock_rag_instance.query.return_value = ("Test response", ["Test source"])
        mock_rag_instance.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }
        mock_session_manager.create_session.return_value = "test_session_123"

        yield mock_rag_instance


@pytest.fixture
def test_app():
    """Create a FastAPI test app that avoids static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create minimal test app without static file mounting
    app = FastAPI(title="Test RAG System")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Mock RAG system for the app
    mock_rag_system = Mock()
    mock_rag_system.session_manager = Mock()

    # API endpoints (inline to avoid import issues)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Attach mock for test access
    app.mock_rag_system = mock_rag_system

    return app


@pytest.fixture
def api_client(test_app):
    """Create a test client for API testing"""
    return TestClient(test_app), test_app.mock_rag_system


@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "title": "Python Programming Basics",
        "instructor": "Dr. Jane Smith",
        "link": "https://example.com/python-course",
        "lessons": [
            {
                "title": "Introduction to Python",
                "content": "Python is a high-level programming language...",
                "lesson_number": 0,
                "link": "https://example.com/lesson-0"
            },
            {
                "title": "Variables and Data Types",
                "content": "In Python, variables are used to store data...",
                "lesson_number": 1,
                "link": "https://example.com/lesson-1"
            }
        ]
    }


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        {
            "content": "Course Python Programming Basics Lesson 0 content: Python is a high-level programming language that emphasizes code readability.",
            "metadata": {
                "course_title": "Python Programming Basics",
                "lesson_title": "Introduction to Python",
                "lesson_number": 0,
                "chunk_index": 0,
                "course_link": "https://example.com/python-course",
                "lesson_link": "https://example.com/lesson-0"
            }
        },
        {
            "content": "Course Python Programming Basics Lesson 1 content: In Python, variables are used to store data values and can be of different types.",
            "metadata": {
                "course_title": "Python Programming Basics",
                "lesson_title": "Variables and Data Types",
                "lesson_number": 1,
                "chunk_index": 0,
                "course_link": "https://example.com/python-course",
                "lesson_link": "https://example.com/lesson-1"
            }
        }
    ]


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    with patch('anthropic.Anthropic') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock messages API
        mock_messages = Mock()
        mock_client.messages = mock_messages

        # Default successful response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test AI response")]
        mock_response.stop_reason = "end_turn"
        mock_messages.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()

    # Mock search results
    mock_store.search.return_value = Mock(
        documents=[["Test document content"]],
        metadatas=[[{"course_title": "Test Course", "lesson_number": 0}]],
        distances=[[0.1]]
    )

    # Mock analytics
    mock_store.get_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }

    yield mock_store


@pytest.fixture
def mock_document_processor():
    """Mock document processor for testing"""
    with patch('document_processor.DocumentProcessor') as mock_processor_class:
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        # Mock successful processing
        mock_processor.process_file.return_value = (
            Mock(title="Test Course"),  # Course object
            [Mock(), Mock()]  # CourseChunk objects
        )

        yield mock_processor


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = [
        "User: Previous question\nAssistant: Previous answer"
    ]
    mock_manager.add_to_conversation.return_value = None

    yield mock_manager


@pytest.fixture
def test_documents_folder(tmp_path):
    """Create a temporary folder with test documents"""
    docs_folder = tmp_path / "test_docs"
    docs_folder.mkdir()

    # Create a sample course document
    course_file = docs_folder / "python_basics.txt"
    course_content = """Course Title: Python Programming Basics
Course Link: https://example.com/python-course
Course Instructor: Dr. Jane Smith

Lesson 0: Introduction to Python
Lesson Link: https://example.com/lesson-0
Python is a high-level programming language that emphasizes code readability and simplicity. It was created by Guido van Rossum and first released in 1991.

Lesson 1: Variables and Data Types
Lesson Link: https://example.com/lesson-1
In Python, variables are used to store data values. Python has several built-in data types including integers, floats, strings, and booleans.
"""

    course_file.write_text(course_content)

    yield str(docs_folder)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Automatically suppress warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client and collections"""
    with patch('chromadb.PersistentClient') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock collections
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []

        # Mock collection methods
        mock_collection.add.return_value = None
        mock_collection.query.return_value = {
            'documents': [['Test document']],
            'metadatas': [[{'course_title': 'Test'}]],
            'distances': [[0.1]]
        }
        mock_collection.count.return_value = 1

        yield mock_client


@pytest.fixture
def env_vars():
    """Set up environment variables for testing"""
    original_env = os.environ.copy()

    # Set test environment variables
    test_env = {
        'ANTHROPIC_API_KEY': 'test_api_key_12345',
        'TESTING': 'true'
    }

    for key, value in test_env.items():
        os.environ[key] = value

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)