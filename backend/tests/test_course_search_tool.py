import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Comprehensive tests for CourseSearchTool.execute() method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = Mock()
        return mock_store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool instance with mocked vector store"""
        tool = CourseSearchTool(mock_vector_store)
        return tool

    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is properly formatted"""
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_successful_search(self, search_tool, mock_vector_store):
        """Test successful search execution with results"""
        # Setup mock search results
        mock_results = SearchResults(
            documents=["Course content about MCP", "More MCP details"],
            metadata=[
                {"course_title": "MCP Introduction", "lesson_number": 1},
                {"course_title": "MCP Introduction", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = mock_results

        # Execute search
        result = search_tool.execute("What is MCP?")

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=None
        )

        # Verify result formatting
        assert "[MCP Introduction - Lesson 1]" in result
        assert "[MCP Introduction - Lesson 2]" in result
        assert "Course content about MCP" in result
        assert "More MCP details" in result

        # Verify sources were tracked
        assert len(search_tool.last_sources) == 2
        assert "MCP Introduction - Lesson 1" in search_tool.last_sources
        assert "MCP Introduction - Lesson 2" in search_tool.last_sources

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test search execution with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("python functions", course_name="Advanced Python")

        mock_vector_store.search.assert_called_once_with(
            query="python functions",
            course_name="Advanced Python",
            lesson_number=None
        )

        assert "[Advanced Python - Lesson 3]" in result
        assert "Filtered content" in result

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test search execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 5}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("machine learning", lesson_number=5)

        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=5
        )

        assert "[Data Science - Lesson 5]" in result
        assert "Lesson specific content" in result

    def test_execute_with_both_filters(self, search_tool, mock_vector_store):
        """Test search execution with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Web Development", "lesson_number": 2}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(
            "HTML basics",
            course_name="Web Development",
            lesson_number=2
        )

        mock_vector_store.search.assert_called_once_with(
            query="HTML basics",
            course_name="Web Development",
            lesson_number=2
        )

        assert "[Web Development - Lesson 2]" in result
        assert "Specific lesson content" in result

    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Test handling of empty search results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("nonexistent topic")

        assert "No relevant content found" in result
        assert search_tool.last_sources == []

    def test_execute_empty_results_with_filters(self, search_tool, mock_vector_store):
        """Test empty results message includes filter information"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(
            "nonexistent",
            course_name="Missing Course",
            lesson_number=99
        )

        assert "No relevant content found in course 'Missing Course' in lesson 99" in result

    def test_execute_search_error(self, search_tool, mock_vector_store):
        """Test handling of search errors from vector store"""
        mock_results = SearchResults.empty("ChromaDB connection failed")
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("test query")

        assert result == "ChromaDB connection failed"
        assert search_tool.last_sources == []

    def test_execute_missing_metadata(self, search_tool, mock_vector_store):
        """Test handling of malformed metadata"""
        mock_results = SearchResults(
            documents=["Content without proper metadata"],
            metadata=[{}],  # Missing course_title and lesson_number
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("test query")

        # Should handle missing metadata gracefully
        assert "[unknown]" in result
        assert "Content without proper metadata" in result

    def test_execute_partial_metadata(self, search_tool, mock_vector_store):
        """Test handling of partial metadata (course title only)"""
        mock_results = SearchResults(
            documents=["Content with partial metadata"],
            metadata=[{"course_title": "Partial Course"}],  # Missing lesson_number
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("test query")

        assert "[Partial Course]" in result  # No lesson number
        assert "Content with partial metadata" in result
        assert "Partial Course" in search_tool.last_sources

    def test_format_results_multiple_documents(self, search_tool):
        """Test formatting of multiple search results"""
        results = SearchResults(
            documents=["First document", "Second document", "Third document"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
                {"course_title": "Course A", "lesson_number": 3}
            ],
            distances=[0.1, 0.2, 0.3]
        )

        formatted = search_tool._format_results(results)

        # Should be separated by double newlines
        sections = formatted.split("\n\n")
        assert len(sections) == 3

        # Check each section has proper header and content
        assert "[Course A - Lesson 1]\nFirst document" in formatted
        assert "[Course B - Lesson 2]\nSecond document" in formatted
        assert "[Course A - Lesson 3]\nThird document" in formatted

    def test_source_tracking_reset(self, search_tool, mock_vector_store):
        """Test that sources are properly tracked and can be reset"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results

        # First search
        search_tool.execute("first query")
        assert len(search_tool.last_sources) == 1

        # Second search should replace sources
        search_tool.execute("second query")
        assert len(search_tool.last_sources) == 1

        # Manual reset
        search_tool.last_sources = []
        assert len(search_tool.last_sources) == 0


class TestToolManager:
    """Tests for ToolManager integration with CourseSearchTool"""

    @pytest.fixture
    def tool_manager(self):
        """Create ToolManager instance"""
        return ToolManager()

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store"""
        return Mock()

    def test_register_tool(self, tool_manager, mock_vector_store):
        """Test tool registration"""
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        assert "search_course_content" in tool_manager.tools

        # Test tool definitions
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, tool_manager, mock_vector_store):
        """Test tool execution through manager"""
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Mock successful search
        mock_results = SearchResults(
            documents=["Test result"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results

        result = tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )

        assert "[Test - Lesson 1]" in result
        assert "Test result" in result

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test execution of non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, tool_manager, mock_vector_store):
        """Test source retrieval from tools"""
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Initially no sources
        assert tool_manager.get_last_sources() == []

        # After search, sources should be available
        search_tool.last_sources = ["Test Course - Lesson 1"]
        sources = tool_manager.get_last_sources()
        assert sources == ["Test Course - Lesson 1"]

    def test_reset_sources(self, tool_manager, mock_vector_store):
        """Test source reset functionality"""
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Set some sources
        search_tool.last_sources = ["Test Course - Lesson 1"]

        # Reset sources
        tool_manager.reset_sources()
        assert search_tool.last_sources == []
        assert tool_manager.get_last_sources() == []


if __name__ == "__main__":
    pytest.main([__file__])