import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from vector_store import SearchResults


class MockConfig:
    """Mock configuration for testing"""

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test_key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    MAX_HISTORY = 2


class TestRAGSystemInitialization:
    """Tests for RAG system initialization and setup"""

    @pytest.fixture
    def mock_config(self):
        return MockConfig()

    def test_rag_system_initialization(self, mock_config):
        """Test that RAG system initializes all components correctly"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool") as mock_search_tool,
        ):

            rag_system = RAGSystem(mock_config)

            # Verify all components were initialized with correct parameters
            mock_doc_proc.assert_called_once_with(
                mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP
            )
            mock_vector_store.assert_called_once_with(
                mock_config.CHROMA_PATH,
                mock_config.EMBEDDING_MODEL,
                mock_config.MAX_RESULTS,
            )
            mock_ai_gen.assert_called_once_with(
                mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
            )
            mock_session_mgr.assert_called_once_with(mock_config.MAX_HISTORY)

            # Verify tool registration
            mock_tool_mgr.return_value.register_tool.assert_called_once()

    def test_search_tool_integration(self, mock_config):
        """Test that search tool is properly integrated"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool") as mock_search_tool,
        ):

            rag_system = RAGSystem(mock_config)

            # Verify search tool was created with vector store
            mock_search_tool.assert_called_once_with(mock_vector_store.return_value)

            # Verify tool was registered
            mock_tool_mgr.return_value.register_tool.assert_called_once_with(
                mock_search_tool.return_value
            )


class TestRAGSystemQuery:
    """Tests for the query processing functionality"""

    @pytest.fixture
    def mock_rag_system(self):
        """Create a RAG system with all components mocked"""
        config = MockConfig()

        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
        ):

            rag_system = RAGSystem(config)

            # Store references to mocked components
            rag_system.mock_ai_generator = mock_ai_gen.return_value
            rag_system.mock_session_manager = mock_session_mgr.return_value
            rag_system.mock_tool_manager = mock_tool_mgr.return_value

            return rag_system

    def test_query_without_session(self, mock_rag_system):
        """Test query processing without session ID"""
        # Setup mocks
        mock_rag_system.mock_ai_generator.generate_response.return_value = "AI response"
        mock_rag_system.mock_tool_manager.get_last_sources.return_value = []

        # Execute query
        response, sources = mock_rag_system.query("What is Python?")

        # Verify AI generator was called correctly
        mock_rag_system.mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system.mock_ai_generator.generate_response.call_args

        assert (
            call_args[1]["query"]
            == "Answer this question about course materials: What is Python?"
        )
        assert call_args[1]["conversation_history"] is None
        assert "tools" in call_args[1]
        assert "tool_manager" in call_args[1]

        # Verify session manager was not called for history
        mock_rag_system.mock_session_manager.get_conversation_history.assert_not_called()

        # Verify sources were retrieved and reset
        mock_rag_system.mock_tool_manager.get_last_sources.assert_called_once()
        mock_rag_system.mock_tool_manager.reset_sources.assert_called_once()

        assert response == "AI response"
        assert sources == []

    def test_query_with_session(self, mock_rag_system):
        """Test query processing with session ID"""
        # Setup mocks
        mock_rag_system.mock_session_manager.get_conversation_history.return_value = (
            "User: Hello\nAssistant: Hi there!"
        )
        mock_rag_system.mock_ai_generator.generate_response.return_value = (
            "AI response with context"
        )
        mock_rag_system.mock_tool_manager.get_last_sources.return_value = [
            "Course 1 - Lesson 1"
        ]

        # Execute query
        response, sources = mock_rag_system.query(
            "Follow up question", session_id="session_1"
        )

        # Verify conversation history was retrieved
        mock_rag_system.mock_session_manager.get_conversation_history.assert_called_once_with(
            "session_1"
        )

        # Verify AI generator received history
        call_args = mock_rag_system.mock_ai_generator.generate_response.call_args
        assert (
            call_args[1]["conversation_history"] == "User: Hello\nAssistant: Hi there!"
        )

        # Verify session was updated
        mock_rag_system.mock_session_manager.add_exchange.assert_called_once_with(
            "session_1", "Follow up question", "AI response with context"
        )

        assert response == "AI response with context"
        assert sources == ["Course 1 - Lesson 1"]

    def test_query_with_sources_tracking(self, mock_rag_system):
        """Test that sources are properly tracked and returned"""
        # Setup mocks with sources
        mock_rag_system.mock_ai_generator.generate_response.return_value = (
            "Response with sources"
        )
        mock_rag_system.mock_tool_manager.get_last_sources.return_value = [
            "Python Course - Lesson 1",
            "Python Course - Lesson 2",
        ]

        # Execute query
        response, sources = mock_rag_system.query("Python basics")

        # Verify sources were retrieved and reset
        mock_rag_system.mock_tool_manager.get_last_sources.assert_called_once()
        mock_rag_system.mock_tool_manager.reset_sources.assert_called_once()

        assert response == "Response with sources"
        assert sources == ["Python Course - Lesson 1", "Python Course - Lesson 2"]

    def test_query_prompt_formatting(self, mock_rag_system):
        """Test that query prompt is properly formatted"""
        mock_rag_system.mock_ai_generator.generate_response.return_value = "response"
        mock_rag_system.mock_tool_manager.get_last_sources.return_value = []

        # Execute query
        mock_rag_system.query("Test query")

        # Verify prompt formatting
        call_args = mock_rag_system.mock_ai_generator.generate_response.call_args
        expected_prompt = "Answer this question about course materials: Test query"
        assert call_args[1]["query"] == expected_prompt

    def test_query_tool_integration(self, mock_rag_system):
        """Test that tools are properly integrated in query"""
        mock_rag_system.mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        mock_rag_system.mock_ai_generator.generate_response.return_value = "response"
        mock_rag_system.mock_tool_manager.get_last_sources.return_value = []

        # Execute query
        mock_rag_system.query("Query with tools")

        # Verify tools were passed to AI generator
        call_args = mock_rag_system.mock_ai_generator.generate_response.call_args
        assert call_args[1]["tools"] == [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        assert call_args[1]["tool_manager"] == mock_rag_system.mock_tool_manager


class TestRAGSystemDocumentProcessing:
    """Tests for document processing functionality"""

    @pytest.fixture
    def rag_system_with_real_components(self):
        """Create RAG system with real document processor but mocked storage"""
        config = MockConfig()

        with (
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.mock_vector_store = mock_vector_store.return_value
            return rag_system

    def test_add_course_document_success(self, rag_system_with_real_components):
        """Test successful course document addition"""
        # Create mock course and chunks
        mock_course = Course(title="Test Course", lessons=[])
        mock_chunks = [
            CourseChunk(content="Chunk 1", course_title="Test Course", chunk_index=0),
            CourseChunk(content="Chunk 2", course_title="Test Course", chunk_index=1),
        ]

        # Mock document processor
        with patch.object(
            rag_system_with_real_components.document_processor,
            "process_course_document",
        ) as mock_process:
            mock_process.return_value = (mock_course, mock_chunks)

            # Execute
            course, chunk_count = rag_system_with_real_components.add_course_document(
                "test_file.txt"
            )

            # Verify processing
            mock_process.assert_called_once_with("test_file.txt")

            # Verify vector store calls
            rag_system_with_real_components.mock_vector_store.add_course_metadata.assert_called_once_with(
                mock_course
            )
            rag_system_with_real_components.mock_vector_store.add_course_content.assert_called_once_with(
                mock_chunks
            )

            assert course == mock_course
            assert chunk_count == 2

    def test_add_course_document_error(self, rag_system_with_real_components):
        """Test handling of document processing errors"""
        # Mock error in document processing
        with patch.object(
            rag_system_with_real_components.document_processor,
            "process_course_document",
        ) as mock_process:
            mock_process.side_effect = Exception("File not found")

            # Execute
            course, chunk_count = rag_system_with_real_components.add_course_document(
                "nonexistent.txt"
            )

            # Verify error handling
            assert course is None
            assert chunk_count == 0

            # Verify vector store was not called
            rag_system_with_real_components.mock_vector_store.add_course_metadata.assert_not_called()
            rag_system_with_real_components.mock_vector_store.add_course_content.assert_not_called()

    def test_add_course_folder_with_existing_courses(
        self, rag_system_with_real_components
    ):
        """Test adding course folder with duplicate detection"""
        # Mock existing course titles
        rag_system_with_real_components.mock_vector_store.get_existing_course_titles.return_value = [
            "Existing Course"
        ]

        # Mock file system
        with (
            patch("rag_system.os.path.exists", return_value=True),
            patch("rag_system.os.listdir", return_value=["course1.txt", "course2.txt"]),
            patch("rag_system.os.path.isfile", return_value=True),
        ):

            # Mock course processing
            new_course = Course(title="New Course", lessons=[])
            existing_course = Course(title="Existing Course", lessons=[])
            mock_chunks = [
                CourseChunk(content="test", course_title="New Course", chunk_index=0)
            ]

            with patch.object(
                rag_system_with_real_components.document_processor,
                "process_course_document",
            ) as mock_process:
                mock_process.side_effect = [
                    (new_course, mock_chunks),  # New course
                    (existing_course, []),  # Existing course
                ]

                # Execute
                total_courses, total_chunks = (
                    rag_system_with_real_components.add_course_folder("test_folder")
                )

                # Verify only new course was added
                assert total_courses == 1
                assert total_chunks == 1

                # Verify vector store calls for new course only
                rag_system_with_real_components.mock_vector_store.add_course_metadata.assert_called_once_with(
                    new_course
                )
                rag_system_with_real_components.mock_vector_store.add_course_content.assert_called_once_with(
                    mock_chunks
                )

    def test_add_course_folder_nonexistent_folder(
        self, rag_system_with_real_components
    ):
        """Test handling of nonexistent folder"""
        with patch("rag_system.os.path.exists", return_value=False):
            courses, chunks = rag_system_with_real_components.add_course_folder(
                "nonexistent"
            )

            assert courses == 0
            assert chunks == 0


class TestRAGSystemAnalytics:
    """Tests for analytics functionality"""

    @pytest.fixture
    def rag_system_analytics(self):
        """Create RAG system for analytics testing"""
        config = MockConfig()

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager"),
            patch("rag_system.CourseSearchTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.mock_vector_store = mock_vector_store.return_value
            return rag_system

    def test_get_course_analytics(self, rag_system_analytics):
        """Test course analytics retrieval"""
        # Mock vector store responses
        rag_system_analytics.mock_vector_store.get_course_count.return_value = 5
        rag_system_analytics.mock_vector_store.get_existing_course_titles.return_value = [
            "Python Basics",
            "JavaScript Fundamentals",
            "Data Science",
            "Machine Learning",
            "Web Development",
        ]

        # Execute
        analytics = rag_system_analytics.get_course_analytics()

        # Verify
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Python Basics" in analytics["course_titles"]


class TestRAGSystemErrorHandling:
    """Tests for error handling in RAG system"""

    @pytest.fixture
    def rag_system_error_test(self):
        """Create RAG system for error testing"""
        config = MockConfig()

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.ToolManager") as mock_tool_mgr,
            patch("rag_system.CourseSearchTool"),
        ):

            rag_system = RAGSystem(config)
            rag_system.mock_ai_generator = mock_ai_gen.return_value
            rag_system.mock_session_manager = mock_session_mgr.return_value
            rag_system.mock_tool_manager = mock_tool_mgr.return_value
            return rag_system

    def test_query_ai_generator_error(self, rag_system_error_test):
        """Test handling of AI generator errors"""
        # Mock AI generator to raise exception
        rag_system_error_test.mock_ai_generator.generate_response.side_effect = (
            Exception("API error")
        )
        rag_system_error_test.mock_tool_manager.get_last_sources.return_value = []

        # This should propagate the exception (as it does in the actual code)
        with pytest.raises(Exception, match="API error"):
            rag_system_error_test.query("test query")

    def test_query_tool_manager_error(self, rag_system_error_test):
        """Test handling of tool manager errors"""
        # Mock tool manager to raise exception
        rag_system_error_test.mock_ai_generator.generate_response.return_value = (
            "response"
        )
        rag_system_error_test.mock_tool_manager.get_last_sources.side_effect = (
            Exception("Tool error")
        )

        # This should propagate the exception
        with pytest.raises(Exception, match="Tool error"):
            rag_system_error_test.query("test query")

    def test_query_session_manager_error(self, rag_system_error_test):
        """Test handling of session manager errors"""
        # Mock session manager to raise exception
        rag_system_error_test.mock_session_manager.get_conversation_history.side_effect = Exception(
            "Session error"
        )

        # This should propagate the exception
        with pytest.raises(Exception, match="Session error"):
            rag_system_error_test.query("test query", session_id="test_session")


if __name__ == "__main__":
    pytest.main([__file__])
