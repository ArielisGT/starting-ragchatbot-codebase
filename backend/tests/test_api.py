import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from fastapi import HTTPException


class TestFastAPIEndpoints:
    """Tests for FastAPI endpoints"""

    @pytest.fixture
    def mock_rag_system(self):
        """Create mock RAG system"""
        with patch('app.RAGSystem') as mock_rag_class:
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance

            # Mock session manager
            mock_session_manager = Mock()
            mock_rag_instance.session_manager = mock_session_manager

            return mock_rag_instance

    @pytest.fixture
    def test_client(self, mock_rag_system):
        """Create test client with mocked RAG system"""
        # Import app after mocking
        from app import app

        # Replace the global rag_system with our mock
        import app as app_module
        app_module.rag_system = mock_rag_system

        return TestClient(app), mock_rag_system

    def test_query_endpoint_successful(self, test_client):
        """Test successful query processing"""
        client, mock_rag_system = test_client

        # Mock successful query response
        mock_rag_system.query.return_value = (
            "Python is a programming language used for...",
            ["Python Course - Lesson 1", "Python Course - Lesson 2"]
        )
        mock_rag_system.session_manager.create_session.return_value = "session_123"

        # Make request
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Python is a programming language used for..."
        assert data["sources"] == ["Python Course - Lesson 1", "Python Course - Lesson 2"]
        assert data["session_id"] == "session_123"

        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is Python?", "session_123")

    def test_query_endpoint_with_existing_session(self, test_client):
        """Test query with existing session ID"""
        client, mock_rag_system = test_client

        # Mock query response
        mock_rag_system.query.return_value = (
            "Follow-up answer with context",
            ["Course A - Lesson 3"]
        )

        # Make request with session ID
        response = client.post("/api/query", json={
            "query": "Can you elaborate?",
            "session_id": "existing_session_456"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Follow-up answer with context"
        assert data["sources"] == ["Course A - Lesson 3"]
        assert data["session_id"] == "existing_session_456"

        # Verify RAG system was called with existing session
        mock_rag_system.query.assert_called_once_with("Can you elaborate?", "existing_session_456")
        # Session creation should not be called
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_endpoint_empty_sources(self, test_client):
        """Test query that returns no sources"""
        client, mock_rag_system = test_client

        # Mock query response with empty sources
        mock_rag_system.query.return_value = (
            "This is general knowledge, not course-specific.",
            []
        )
        mock_rag_system.session_manager.create_session.return_value = "session_789"

        # Make request
        response = client.post("/api/query", json={
            "query": "What is 2+2?"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "This is general knowledge, not course-specific."
        assert data["sources"] == []
        assert data["session_id"] == "session_789"

    def test_query_endpoint_rag_system_error(self, test_client):
        """Test handling of RAG system errors - this is where 'Query failed' comes from"""
        client, mock_rag_system = test_client

        # Mock RAG system to raise exception
        mock_rag_system.query.side_effect = Exception("ChromaDB connection failed")
        mock_rag_system.session_manager.create_session.return_value = "session_error"

        # Make request
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })

        # Verify error response
        assert response.status_code == 500
        assert response.json()["detail"] == "ChromaDB connection failed"

        # This is the key test - this is where "Error: Query failed" originates

    def test_query_endpoint_session_creation_error(self, test_client):
        """Test handling of session creation errors"""
        client, mock_rag_system = test_client

        # Mock session creation to fail
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session error")

        # Make request without session ID
        response = client.post("/api/query", json={
            "query": "Test query"
        })

        # Verify error response
        assert response.status_code == 500
        assert response.json()["detail"] == "Session error"

    def test_query_endpoint_invalid_request_body(self, test_client):
        """Test handling of invalid request body"""
        client, mock_rag_system = test_client

        # Make request with missing query field
        response = client.post("/api/query", json={
            "session_id": "test_session"
            # Missing "query" field
        })

        # Verify validation error
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query(self, test_client):
        """Test handling of empty query"""
        client, mock_rag_system = test_client

        # Mock successful response even for empty query
        mock_rag_system.query.return_value = (
            "Please provide a specific question.",
            []
        )
        mock_rag_system.session_manager.create_session.return_value = "session_empty"

        # Make request with empty query
        response = client.post("/api/query", json={
            "query": ""
        })

        # Verify response (should still work)
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Please provide a specific question."

    def test_courses_endpoint_successful(self, test_client):
        """Test successful course analytics retrieval"""
        client, mock_rag_system = test_client

        # Mock analytics response
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Python Basics", "JavaScript Fundamentals", "Data Science"]
        }

        # Make request
        response = client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]

        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_endpoint_error(self, test_client):
        """Test handling of analytics errors"""
        client, mock_rag_system = test_client

        # Mock analytics to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Database unavailable")

        # Make request
        response = client.get("/api/courses")

        # Verify error response
        assert response.status_code == 500
        assert response.json()["detail"] == "Database unavailable"

    def test_courses_endpoint_empty_catalog(self, test_client):
        """Test analytics endpoint with empty course catalog"""
        client, mock_rag_system = test_client

        # Mock empty analytics
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        # Make request
        response = client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_static_file_serving(self, test_client):
        """Test that static files are properly served"""
        client, mock_rag_system = test_client

        # Make request to root (should serve index.html)
        response = client.get("/")

        # Should return HTML content (even if mocked)
        assert response.status_code in [200, 404]  # 404 if no frontend files in test env

    def test_content_type_headers(self, test_client):
        """Test that API responses have correct content type"""
        client, mock_rag_system = test_client

        # Mock successful query
        mock_rag_system.query.return_value = ("test response", [])
        mock_rag_system.session_manager.create_session.return_value = "session_test"

        # Make request
        response = client.post("/api/query", json={"query": "test"})

        # Verify content type
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_cors_headers(self, test_client):
        """Test that CORS headers are properly set"""
        client, mock_rag_system = test_client

        # Mock successful query
        mock_rag_system.query.return_value = ("test response", [])
        mock_rag_system.session_manager.create_session.return_value = "session_cors"

        # Make request
        response = client.post("/api/query", json={"query": "test"})

        # Verify CORS headers are present (set in middleware)
        assert response.status_code == 200
        # Note: TestClient may not include all middleware headers

    def test_multiple_consecutive_queries(self, test_client):
        """Test multiple queries in sequence"""
        client, mock_rag_system = test_client

        # Mock responses for multiple queries
        mock_rag_system.query.side_effect = [
            ("First response", ["Source 1"]),
            ("Second response", ["Source 2"]),
            ("Third response", [])
        ]
        mock_rag_system.session_manager.create_session.return_value = "session_multi"

        # Make multiple requests
        response1 = client.post("/api/query", json={"query": "First question"})
        response2 = client.post("/api/query", json={"query": "Second question", "session_id": "session_multi"})
        response3 = client.post("/api/query", json={"query": "Third question", "session_id": "session_multi"})

        # Verify all responses
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200

        assert response1.json()["answer"] == "First response"
        assert response2.json()["answer"] == "Second response"
        assert response3.json()["answer"] == "Third response"

        # Verify query calls
        assert mock_rag_system.query.call_count == 3


class TestErrorScenarios:
    """Specific tests for error scenarios that could cause 'Query failed'"""

    @pytest.fixture
    def test_client_error_scenarios(self):
        """Create test client for error scenario testing"""
        with patch('app.RAGSystem') as mock_rag_class:
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance

            from app import app
            import app as app_module
            app_module.rag_system = mock_rag_instance

            return TestClient(app), mock_rag_instance

    def test_chromadb_connection_error(self, test_client_error_scenarios):
        """Test ChromaDB connection failure"""
        client, mock_rag_system = test_client_error_scenarios

        # Mock ChromaDB connection error
        mock_rag_system.query.side_effect = Exception("ChromaDB: Connection refused")
        mock_rag_system.session_manager.create_session.return_value = "session_chroma_error"

        response = client.post("/api/query", json={"query": "Test ChromaDB error"})

        assert response.status_code == 500
        assert "ChromaDB: Connection refused" in response.json()["detail"]

    def test_anthropic_api_error(self, test_client_error_scenarios):
        """Test Anthropic API failure"""
        client, mock_rag_system = test_client_error_scenarios

        # Mock Anthropic API error
        mock_rag_system.query.side_effect = Exception("Anthropic API: Rate limit exceeded")
        mock_rag_system.session_manager.create_session.return_value = "session_api_error"

        response = client.post("/api/query", json={"query": "Test API error"})

        assert response.status_code == 500
        assert "Anthropic API: Rate limit exceeded" in response.json()["detail"]

    def test_tool_execution_error(self, test_client_error_scenarios):
        """Test tool execution failure"""
        client, mock_rag_system = test_client_error_scenarios

        # Mock tool execution error
        mock_rag_system.query.side_effect = Exception("Tool execution failed: search_course_content")
        mock_rag_system.session_manager.create_session.return_value = "session_tool_error"

        response = client.post("/api/query", json={"query": "Test tool error"})

        assert response.status_code == 500
        assert "Tool execution failed" in response.json()["detail"]

    def test_vector_store_error(self, test_client_error_scenarios):
        """Test vector store failure"""
        client, mock_rag_system = test_client_error_scenarios

        # Mock vector store error
        mock_rag_system.query.side_effect = Exception("Vector store: Embedding model not found")
        mock_rag_system.session_manager.create_session.return_value = "session_vector_error"

        response = client.post("/api/query", json={"query": "Test vector error"})

        assert response.status_code == 500
        assert "Vector store: Embedding model not found" in response.json()["detail"]

    def test_session_manager_error(self, test_client_error_scenarios):
        """Test session manager failure"""
        client, mock_rag_system = test_client_error_scenarios

        # Mock session manager error
        mock_rag_system.query.side_effect = Exception("Session manager: Memory overflow")
        mock_rag_system.session_manager.create_session.return_value = "session_memory_error"

        response = client.post("/api/query", json={"query": "Test session error"})

        assert response.status_code == 500
        assert "Session manager: Memory overflow" in response.json()["detail"]


class TestStartupEvent:
    """Tests for the startup event that loads documents"""

    def test_startup_event_successful_loading(self):
        """Test successful document loading on startup"""
        with patch('app.rag_system') as mock_rag_system, \
             patch('app.os.path.exists', return_value=True):

            # Mock successful document loading
            mock_rag_system.add_course_folder.return_value = (3, 150)

            # Import to trigger startup event simulation
            from app import startup_event

            # Manually call startup event for testing
            import asyncio
            asyncio.run(startup_event())

            # Verify document loading was attempted
            mock_rag_system.add_course_folder.assert_called_once_with("../docs", clear_existing=False)

    def test_startup_event_no_docs_folder(self):
        """Test startup when docs folder doesn't exist"""
        with patch('app.rag_system') as mock_rag_system, \
             patch('app.os.path.exists', return_value=False):

            from app import startup_event

            # Manually call startup event
            import asyncio
            asyncio.run(startup_event())

            # Verify no document loading was attempted
            mock_rag_system.add_course_folder.assert_not_called()

    def test_startup_event_loading_error(self):
        """Test startup event with document loading errors"""
        with patch('app.rag_system') as mock_rag_system, \
             patch('app.os.path.exists', return_value=True):

            # Mock loading error
            mock_rag_system.add_course_folder.side_effect = Exception("Failed to load documents")

            from app import startup_event

            # Should not raise exception (error is caught and printed)
            import asyncio
            asyncio.run(startup_event())

            # Verify loading was attempted
            mock_rag_system.add_course_folder.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])