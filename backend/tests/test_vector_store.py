import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Tests for SearchResults class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course_title': 'Course1'}, {'course_title': 'Course2'}]],
            'distances': [[0.1, 0.2]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course_title': 'Course1'}, {'course_title': 'Course2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    def test_from_chroma_no_results(self):
        """Test creating SearchResults from ChromaDB with no results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_results_with_error(self):
        """Test creating empty results with error message"""
        results = SearchResults.empty("Database connection failed")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Database connection failed"

    def test_is_empty(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()

        # Non-empty results
        results_with_data = SearchResults(['doc1'], [{'title': 'test'}], [0.1])
        assert not results_with_data.is_empty()


class TestVectorStore:
    """Tests for VectorStore class"""

    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Mock collections
            mock_catalog = Mock()
            mock_content = Mock()
            mock_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]

            yield mock_instance, mock_catalog, mock_content

    @pytest.fixture
    def vector_store(self, mock_chroma_client):
        """Create VectorStore instance with mocked ChromaDB"""
        mock_client, mock_catalog, mock_content = mock_chroma_client

        with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            store = VectorStore("./test_db", "test-model", max_results=3)
            store.client = mock_client
            store.course_catalog = mock_catalog
            store.course_content = mock_content
            return store, mock_catalog, mock_content

    def test_vector_store_initialization(self):
        """Test VectorStore initialization"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_embedding:

            mock_instance = Mock()
            mock_client.return_value = mock_instance

            store = VectorStore("./test_db", "test-model", max_results=5)

            # Verify client initialization
            mock_client.assert_called_once()
            mock_embedding.assert_called_once_with(model_name="test-model")

            # Verify collections were created
            assert mock_instance.get_or_create_collection.call_count == 2

            assert store.max_results == 5

    def test_search_successful(self, vector_store):
        """Test successful search operation"""
        store, mock_catalog, mock_content = vector_store

        # Mock search results
        mock_content.query.return_value = {
            'documents': [['Test document content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }

        # Execute search
        results = store.search("test query")

        # Verify query was called correctly
        mock_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            where=None
        )

        # Verify results
        assert len(results.documents) == 1
        assert results.documents[0] == "Test document content"
        assert results.metadata[0]['course_title'] == 'Test Course'
        assert results.error is None

    def test_search_with_course_filter(self, vector_store):
        """Test search with course name filter"""
        store, mock_catalog, mock_content = vector_store

        # Mock course resolution
        mock_catalog.query.return_value = {
            'documents': [['Python Basics Course']],
            'metadatas': [[{'title': 'Python Basics'}]],
            'distances': [[0.05]]
        }

        # Mock content search
        mock_content.query.return_value = {
            'documents': [['Python content']],
            'metadatas': [[{'course_title': 'Python Basics'}]],
            'distances': [[0.1]]
        }

        # Execute search with course filter
        results = store.search("functions", course_name="Python")

        # Verify course resolution was called
        mock_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )

        # Verify content search with filter
        mock_content.query.assert_called_once_with(
            query_texts=["functions"],
            n_results=3,
            where={"course_title": "Python Basics"}
        )

        assert len(results.documents) == 1
        assert results.documents[0] == "Python content"

    def test_search_with_lesson_filter(self, vector_store):
        """Test search with lesson number filter"""
        store, mock_catalog, mock_content = vector_store

        # Mock content search
        mock_content.query.return_value = {
            'documents': [['Lesson content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 5}]],
            'distances': [[0.1]]
        }

        # Execute search with lesson filter
        results = store.search("test query", lesson_number=5)

        # Verify content search with lesson filter
        mock_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            where={"lesson_number": 5}
        )

        assert len(results.documents) == 1

    def test_search_with_both_filters(self, vector_store):
        """Test search with both course and lesson filters"""
        store, mock_catalog, mock_content = vector_store

        # Mock course resolution
        mock_catalog.query.return_value = {
            'documents': [['Course']],
            'metadatas': [[{'title': 'Advanced Python'}]],
            'distances': [[0.05]]
        }

        # Mock content search
        mock_content.query.return_value = {
            'documents': [['Specific content']],
            'metadatas': [[{'course_title': 'Advanced Python', 'lesson_number': 3}]],
            'distances': [[0.1]]
        }

        # Execute search with both filters
        results = store.search("decorators", course_name="Advanced Python", lesson_number=3)

        # Verify filter combination
        mock_content.query.assert_called_once_with(
            query_texts=["decorators"],
            n_results=3,
            where={"$and": [
                {"course_title": "Advanced Python"},
                {"lesson_number": 3}
            ]}
        )

        assert len(results.documents) == 1

    def test_search_course_not_found(self, vector_store):
        """Test search when course is not found"""
        store, mock_catalog, mock_content = vector_store

        # Mock empty course resolution
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        # Execute search
        results = store.search("test", course_name="Nonexistent Course")

        # Verify error is returned
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()

        # Verify content search was not called
        mock_content.query.assert_not_called()

    def test_search_chromadb_error(self, vector_store):
        """Test handling of ChromaDB errors"""
        store, mock_catalog, mock_content = vector_store

        # Mock ChromaDB exception
        mock_content.query.side_effect = Exception("Database connection failed")

        # Execute search
        results = store.search("test query")

        # Verify error handling
        assert results.error == "Search error: Database connection failed"
        assert results.is_empty()

    def test_resolve_course_name_successful(self, vector_store):
        """Test successful course name resolution"""
        store, mock_catalog, mock_content = vector_store

        # Mock successful resolution
        mock_catalog.query.return_value = {
            'documents': [['Python Programming Course']],
            'metadatas': [[{'title': 'Python Programming'}]],
            'distances': [[0.1]]
        }

        # Test resolution
        resolved_title = store._resolve_course_name("Python")

        assert resolved_title == "Python Programming"

    def test_resolve_course_name_error(self, vector_store):
        """Test course name resolution with error"""
        store, mock_catalog, mock_content = vector_store

        # Mock exception
        mock_catalog.query.side_effect = Exception("Query failed")

        # Test resolution
        resolved_title = store._resolve_course_name("Python")

        assert resolved_title is None

    def test_build_filter_combinations(self, vector_store):
        """Test filter building logic"""
        store, mock_catalog, mock_content = vector_store

        # Test no filters
        filter_dict = store._build_filter(None, None)
        assert filter_dict is None

        # Test course only
        filter_dict = store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

        # Test lesson only
        filter_dict = store._build_filter(None, 5)
        assert filter_dict == {"lesson_number": 5}

        # Test both filters
        filter_dict = store._build_filter("Test Course", 5)
        assert filter_dict == {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 5}
        ]}

    def test_add_course_metadata(self, vector_store):
        """Test adding course metadata"""
        store, mock_catalog, mock_content = vector_store

        # Create test course
        lessons = [
            Lesson(lesson_number=1, title="Introduction", lesson_link="http://lesson1.com"),
            Lesson(lesson_number=2, title="Basics", lesson_link="http://lesson2.com")
        ]
        course = Course(
            title="Test Course",
            course_link="http://course.com",
            instructor="Test Instructor",
            lessons=lessons
        )

        # Add metadata
        store.add_course_metadata(course)

        # Verify catalog was called correctly
        mock_catalog.add.assert_called_once()
        call_args = mock_catalog.add.call_args

        # Check documents
        assert call_args[1]["documents"] == ["Test Course"]
        assert call_args[1]["ids"] == ["Test Course"]

        # Check metadata structure
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Test Course"
        assert metadata["instructor"] == "Test Instructor"
        assert metadata["course_link"] == "http://course.com"
        assert metadata["lesson_count"] == 2

        # Check lessons JSON
        import json
        lessons_data = json.loads(metadata["lessons_json"])
        assert len(lessons_data) == 2
        assert lessons_data[0]["lesson_number"] == 1
        assert lessons_data[0]["lesson_title"] == "Introduction"

    def test_add_course_content(self, vector_store):
        """Test adding course content chunks"""
        store, mock_catalog, mock_content = vector_store

        # Create test chunks
        chunks = [
            CourseChunk(
                content="First chunk content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Second chunk content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            )
        ]

        # Add content
        store.add_course_content(chunks)

        # Verify content collection was called
        mock_content.add.assert_called_once()
        call_args = mock_content.add.call_args

        # Check documents
        assert call_args[1]["documents"] == ["First chunk content", "Second chunk content"]

        # Check metadata
        metadata = call_args[1]["metadatas"]
        assert len(metadata) == 2
        assert metadata[0]["course_title"] == "Test Course"
        assert metadata[0]["lesson_number"] == 1
        assert metadata[0]["chunk_index"] == 0

        # Check IDs
        assert call_args[1]["ids"] == ["Test_Course_0", "Test_Course_1"]

    def test_add_course_content_empty(self, vector_store):
        """Test adding empty course content"""
        store, mock_catalog, mock_content = vector_store

        # Add empty chunks
        store.add_course_content([])

        # Verify no calls were made
        mock_content.add.assert_not_called()

    def test_clear_all_data(self, vector_store):
        """Test clearing all data"""
        store, mock_catalog, mock_content = vector_store

        # Mock collection recreation
        new_catalog = Mock()
        new_content = Mock()
        store.client.get_or_create_collection.side_effect = [new_catalog, new_content]

        # Clear data
        store.clear_all_data()

        # Verify collections were deleted
        store.client.delete_collection.assert_any_call("course_catalog")
        store.client.delete_collection.assert_any_call("course_content")

        # Verify collections were recreated
        assert store.client.get_or_create_collection.call_count == 2

    def test_get_existing_course_titles(self, vector_store):
        """Test getting existing course titles"""
        store, mock_catalog, mock_content = vector_store

        # Mock catalog response
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }

        # Get titles
        titles = store.get_existing_course_titles()

        # Verify
        assert titles == ['Course 1', 'Course 2', 'Course 3']

    def test_get_existing_course_titles_error(self, vector_store):
        """Test error handling in get_existing_course_titles"""
        store, mock_catalog, mock_content = vector_store

        # Mock exception
        mock_catalog.get.side_effect = Exception("Database error")

        # Get titles
        titles = store.get_existing_course_titles()

        # Verify empty list is returned
        assert titles == []

    def test_get_course_count(self, vector_store):
        """Test getting course count"""
        store, mock_catalog, mock_content = vector_store

        # Mock catalog response
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }

        # Get count
        count = store.get_course_count()

        # Verify
        assert count == 2

    def test_get_course_count_error(self, vector_store):
        """Test error handling in get_course_count"""
        store, mock_catalog, mock_content = vector_store

        # Mock exception
        mock_catalog.get.side_effect = Exception("Database error")

        # Get count
        count = store.get_course_count()

        # Verify zero is returned
        assert count == 0

    def test_search_with_custom_limit(self, vector_store):
        """Test search with custom result limit"""
        store, mock_catalog, mock_content = vector_store

        # Mock content search
        mock_content.query.return_value = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{}, {}]],
            'distances': [[0.1, 0.2]]
        }

        # Execute search with custom limit
        results = store.search("test query", limit=10)

        # Verify custom limit was used
        mock_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=10,
            where=None
        )

    def test_get_course_link(self, vector_store):
        """Test getting course link by title"""
        store, mock_catalog, mock_content = vector_store

        # Mock catalog response
        mock_catalog.get.return_value = {
            'metadatas': [{'course_link': 'http://example.com/course'}]
        }

        # Get course link
        link = store.get_course_link("Test Course")

        # Verify
        mock_catalog.get.assert_called_once_with(ids=["Test Course"])
        assert link == "http://example.com/course"

    def test_get_lesson_link(self, vector_store):
        """Test getting lesson link by course and lesson number"""
        store, mock_catalog, mock_content = vector_store

        # Mock catalog response with lessons JSON
        lessons_json = '[{"lesson_number": 1, "lesson_link": "http://lesson1.com"}, {"lesson_number": 2, "lesson_link": "http://lesson2.com"}]'
        mock_catalog.get.return_value = {
            'metadatas': [{'lessons_json': lessons_json}]
        }

        # Get lesson link
        link = store.get_lesson_link("Test Course", 2)

        # Verify
        mock_catalog.get.assert_called_once_with(ids=["Test Course"])
        assert link == "http://lesson2.com"


if __name__ == "__main__":
    pytest.main([__file__])