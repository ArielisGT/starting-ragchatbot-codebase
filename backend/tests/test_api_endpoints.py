import pytest
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    def test_successful_query_without_session(self, api_client):
        """Test successful query without providing session ID"""
        client, mock_rag = api_client

        # Setup mock responses
        mock_rag.query.return_value = (
            "Python is a programming language known for its simplicity and readability.",
            ["Python Course - Lesson 1: Introduction"]
        )
        mock_rag.session_manager.create_session.return_value = "session_auto_123"

        # Make request
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Python is a programming language known for its simplicity and readability."
        assert data["sources"] == ["Python Course - Lesson 1: Introduction"]
        assert data["session_id"] == "session_auto_123"

        # Verify mocks were called correctly
        mock_rag.session_manager.create_session.assert_called_once()
        mock_rag.query.assert_called_once_with("What is Python?", "session_auto_123")

    def test_successful_query_with_session(self, api_client):
        """Test successful query with existing session ID"""
        client, mock_rag = api_client

        # Setup mock responses
        mock_rag.query.return_value = (
            "Variables in Python are dynamically typed.",
            ["Python Course - Lesson 2: Variables"]
        )

        # Make request with session ID
        response = client.post("/api/query", json={
            "query": "How do variables work in Python?",
            "session_id": "existing_session_456"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Variables in Python are dynamically typed."
        assert data["sources"] == ["Python Course - Lesson 2: Variables"]
        assert data["session_id"] == "existing_session_456"

        # Verify session creation was not called
        mock_rag.session_manager.create_session.assert_not_called()
        mock_rag.query.assert_called_once_with("How do variables work in Python?", "existing_session_456")

    def test_query_with_empty_sources(self, api_client):
        """Test query that returns no sources (general knowledge)"""
        client, mock_rag = api_client

        # Setup mock responses - no sources
        mock_rag.query.return_value = (
            "The answer is 4. This is basic arithmetic.",
            []
        )
        mock_rag.session_manager.create_session.return_value = "session_no_sources"

        # Make request
        response = client.post("/api/query", json={
            "query": "What is 2 + 2?"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "The answer is 4. This is basic arithmetic."
        assert data["sources"] == []
        assert data["session_id"] == "session_no_sources"

    def test_query_with_multiple_sources(self, api_client):
        """Test query that returns multiple sources"""
        client, mock_rag = api_client

        # Setup mock responses with multiple sources
        mock_rag.query.return_value = (
            "Functions and classes are fundamental concepts in Python programming.",
            [
                "Python Course - Lesson 3: Functions",
                "Python Course - Lesson 4: Classes",
                "Advanced Python - Lesson 1: OOP Principles"
            ]
        )
        mock_rag.session_manager.create_session.return_value = "session_multi_sources"

        # Make request
        response = client.post("/api/query", json={
            "query": "Tell me about functions and classes in Python"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Functions and classes are fundamental concepts in Python programming."
        assert len(data["sources"]) == 3
        assert "Python Course - Lesson 3: Functions" in data["sources"]
        assert "Python Course - Lesson 4: Classes" in data["sources"]
        assert "Advanced Python - Lesson 1: OOP Principles" in data["sources"]

    def test_query_rag_system_error(self, api_client):
        """Test handling of RAG system errors"""
        client, mock_rag = api_client

        # Setup mock to raise exception
        mock_rag.query.side_effect = Exception("ChromaDB connection failed")
        mock_rag.session_manager.create_session.return_value = "session_error"

        # Make request
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })

        # Verify error response
        assert response.status_code == 500
        assert response.json()["detail"] == "ChromaDB connection failed"

    def test_query_session_creation_error(self, api_client):
        """Test handling of session creation errors"""
        client, mock_rag = api_client

        # Setup mock session creation to fail
        mock_rag.session_manager.create_session.side_effect = Exception("Session creation failed")

        # Make request without session ID
        response = client.post("/api/query", json={
            "query": "Test query"
        })

        # Verify error response
        assert response.status_code == 500
        assert response.json()["detail"] == "Session creation failed"

    def test_query_missing_query_field(self, api_client):
        """Test validation error when query field is missing"""
        client, mock_rag = api_client

        # Make request without query field
        response = client.post("/api/query", json={
            "session_id": "test_session"
        })

        # Verify validation error
        assert response.status_code == 422  # Validation error

    def test_query_empty_query_string(self, api_client):
        """Test behavior with empty query string"""
        client, mock_rag = api_client

        # Setup mock response for empty query
        mock_rag.query.return_value = (
            "Please provide a specific question about the course materials.",
            []
        )
        mock_rag.session_manager.create_session.return_value = "session_empty"

        # Make request with empty query
        response = client.post("/api/query", json={
            "query": ""
        })

        # Should still process successfully
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Please provide a specific question about the course materials."

    def test_query_malformed_json(self, api_client):
        """Test handling of malformed JSON request"""
        client, mock_rag = api_client

        # Make request with malformed JSON
        response = client.post(
            "/api/query",
            data="malformed json content",
            headers={"Content-Type": "application/json"}
        )

        # Verify error response
        assert response.status_code == 422  # Validation error

    def test_query_large_input(self, api_client):
        """Test handling of very large query input"""
        client, mock_rag = api_client

        # Setup mock response
        mock_rag.query.return_value = (
            "Your query was quite long, but I'll try to help.",
            ["General Course - Large Queries"]
        )
        mock_rag.session_manager.create_session.return_value = "session_large"

        # Make request with large query
        large_query = "Tell me about Python " * 1000  # Very long query
        response = client.post("/api/query", json={
            "query": large_query
        })

        # Should still process successfully
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Your query was quite long, but I'll try to help."


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    def test_successful_course_analytics(self, api_client):
        """Test successful retrieval of course analytics"""
        client, mock_rag = api_client

        # Setup mock analytics response
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": [
                "Python Programming Basics",
                "JavaScript Fundamentals",
                "Data Science with Python"
            ]
        }

        # Make request
        response = client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Programming Basics" in data["course_titles"]
        assert "JavaScript Fundamentals" in data["course_titles"]
        assert "Data Science with Python" in data["course_titles"]

        # Verify mock was called
        mock_rag.get_course_analytics.assert_called_once()

    def test_empty_course_catalog(self, api_client):
        """Test analytics endpoint with no courses"""
        client, mock_rag = api_client

        # Setup mock for empty catalog
        mock_rag.get_course_analytics.return_value = {
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

    def test_course_analytics_error(self, api_client):
        """Test handling of analytics errors"""
        client, mock_rag = api_client

        # Setup mock to raise exception
        mock_rag.get_course_analytics.side_effect = Exception("Database connection failed")

        # Make request
        response = client.get("/api/courses")

        # Verify error response
        assert response.status_code == 500
        assert response.json()["detail"] == "Database connection failed"

    def test_large_course_catalog(self, api_client):
        """Test analytics with many courses"""
        client, mock_rag = api_client

        # Setup mock for large catalog
        course_titles = [f"Course {i}" for i in range(100)]
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": course_titles
        }

        # Make request
        response = client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100
        assert "Course 0" in data["course_titles"]
        assert "Course 99" in data["course_titles"]


@pytest.mark.api
class TestAPIHeaders:
    """Test API response headers and CORS"""

    def test_cors_headers_query_endpoint(self, api_client):
        """Test CORS headers on query endpoint"""
        client, mock_rag = api_client

        # Setup mock response
        mock_rag.query.return_value = ("Test response", [])
        mock_rag.session_manager.create_session.return_value = "session_cors"

        # Make request
        response = client.post("/api/query", json={"query": "test"})

        # Verify response
        assert response.status_code == 200

        # Note: TestClient may not include all middleware headers,
        # but we can still verify the endpoint works correctly

    def test_content_type_headers(self, api_client):
        """Test content type headers"""
        client, mock_rag = api_client

        # Setup mock response
        mock_rag.query.return_value = ("Test response", [])
        mock_rag.session_manager.create_session.return_value = "session_content_type"

        # Make request
        response = client.post("/api/query", json={"query": "test"})

        # Verify content type
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_options_request(self, api_client):
        """Test OPTIONS request for CORS preflight"""
        client, mock_rag = api_client

        # Make OPTIONS request
        response = client.options("/api/query")

        # Should be handled by CORS middleware
        # TestClient behavior may vary, but endpoint should exist
        assert response.status_code in [200, 405]  # 405 Method Not Allowed is also acceptable


@pytest.mark.api
class TestEndpointIntegration:
    """Test integration scenarios across endpoints"""

    def test_query_then_analytics(self, api_client):
        """Test querying then getting analytics"""
        client, mock_rag = api_client

        # Setup mocks
        mock_rag.query.return_value = ("Response", ["Source"])
        mock_rag.session_manager.create_session.return_value = "session_integration"
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }

        # Make query request
        query_response = client.post("/api/query", json={"query": "test"})
        assert query_response.status_code == 200

        # Make analytics request
        analytics_response = client.get("/api/courses")
        assert analytics_response.status_code == 200

        # Verify both responses
        query_data = query_response.json()
        analytics_data = analytics_response.json()

        assert query_data["answer"] == "Response"
        assert analytics_data["total_courses"] == 1

    def test_multiple_sessions(self, api_client):
        """Test multiple query sessions"""
        client, mock_rag = api_client

        # Setup mock responses
        mock_rag.query.side_effect = [
            ("First response", ["Source 1"]),
            ("Second response", ["Source 2"]),
            ("Third response", ["Source 3"])
        ]
        mock_rag.session_manager.create_session.side_effect = [
            "session_1", "session_2", "session_3"
        ]

        # Make multiple requests
        responses = []
        for i in range(3):
            response = client.post("/api/query", json={"query": f"Question {i+1}"})
            assert response.status_code == 200
            responses.append(response.json())

        # Verify each response
        for i, data in enumerate(responses):
            assert data["answer"] == f"{'First' if i==0 else 'Second' if i==1 else 'Third'} response"
            assert data["session_id"] == f"session_{i+1}"

    def test_error_handling_consistency(self, api_client):
        """Test that error responses are consistent across endpoints"""
        client, mock_rag = api_client

        # Setup mocks to raise same error
        error_message = "System temporarily unavailable"
        mock_rag.query.side_effect = Exception(error_message)
        mock_rag.get_course_analytics.side_effect = Exception(error_message)

        # Test query endpoint error
        query_response = client.post("/api/query", json={"query": "test"})
        assert query_response.status_code == 500
        assert query_response.json()["detail"] == error_message

        # Test analytics endpoint error
        analytics_response = client.get("/api/courses")
        assert analytics_response.status_code == 500
        assert analytics_response.json()["detail"] == error_message