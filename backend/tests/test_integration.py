import pytest
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem


class TestRealSystemIntegration:
    """Integration tests with real system components to identify actual failures"""

    @pytest.fixture
    def real_rag_system(self):
        """Create RAG system with real components (except API key)"""
        # Override API key for testing (empty key will cause API failures, not system failures)
        test_config = config
        test_config.ANTHROPIC_API_KEY = "test_key"
        return RAGSystem(test_config)

    def test_system_initialization(self, real_rag_system):
        """Test that the system can initialize without errors"""
        # This should work even without a valid API key
        assert real_rag_system is not None
        assert real_rag_system.vector_store is not None
        assert real_rag_system.ai_generator is not None
        assert real_rag_system.tool_manager is not None
        assert real_rag_system.search_tool is not None

    def test_tool_definitions(self, real_rag_system):
        """Test that tools are properly registered and defined"""
        tool_definitions = real_rag_system.tool_manager.get_tool_definitions()

        assert len(tool_definitions) == 1
        assert tool_definitions[0]["name"] == "search_course_content"
        assert "description" in tool_definitions[0]
        assert "input_schema" in tool_definitions[0]

    def test_vector_store_basic_operations(self, real_rag_system):
        """Test vector store basic operations without data"""
        # Test getting course count (should be 0 for empty database)
        count = real_rag_system.vector_store.get_course_count()
        assert count >= 0  # Should not error

        # Test getting course titles (should be empty list)
        titles = real_rag_system.vector_store.get_existing_course_titles()
        assert isinstance(titles, list)

    def test_search_with_empty_database(self, real_rag_system):
        """Test search against empty vector database"""
        try:
            # This should not crash, but may return no results
            results = real_rag_system.vector_store.search("test query")

            # Should return SearchResults object
            assert hasattr(results, 'documents')
            assert hasattr(results, 'metadata')
            assert hasattr(results, 'error')

            # Either successful empty results or error message
            if results.error:
                print(f"Search error (expected for empty DB): {results.error}")
            else:
                assert results.is_empty()  # Should be empty for empty database

        except Exception as e:
            # If there's an exception, it might be our root cause
            print(f"Search failed with exception: {e}")
            raise

    def test_course_search_tool_with_empty_database(self, real_rag_system):
        """Test CourseSearchTool execution with empty database"""
        try:
            # Execute search tool directly
            result = real_rag_system.search_tool.execute("What is Python?")

            # Should return string result, not error
            assert isinstance(result, str)

            # For empty database, should say no content found
            assert "No relevant content found" in result or "error" in result.lower()

        except Exception as e:
            print(f"CourseSearchTool failed with exception: {e}")
            raise

    def test_ai_generator_without_api_key(self, real_rag_system):
        """Test AI generator behavior without valid API key"""
        try:
            # This should fail at the API call level, not at the system level
            real_rag_system.ai_generator.generate_response("test query")

        except Exception as e:
            # Expected to fail due to invalid API key
            print(f"AI Generator failed as expected (invalid API key): {e}")
            # Check if it's an authentication error vs system error
            assert "api" in str(e).lower() or "auth" in str(e).lower() or "key" in str(e).lower()

    def test_rag_system_query_without_api_key(self, real_rag_system):
        """Test full RAG system query without valid API key"""
        try:
            # This is the key test - this is where the "Query failed" error would originate
            response, sources = real_rag_system.query("What is Python programming?")

            # If this succeeds, there's no system-level issue
            assert isinstance(response, str)
            assert isinstance(sources, list)

        except Exception as e:
            # This is likely where the "Query failed" error comes from
            print(f"RAG System query failed: {type(e).__name__}: {e}")

            # Check what type of error this is
            error_str = str(e).lower()

            if "api" in error_str or "auth" in error_str or "key" in error_str:
                print("Error is related to API authentication (expected)")
            elif "chroma" in error_str or "database" in error_str:
                print("Error is related to ChromaDB (potential root cause)")
            elif "embedding" in error_str or "model" in error_str:
                print("Error is related to embedding model (potential root cause)")
            elif "tool" in error_str:
                print("Error is related to tool execution (potential root cause)")
            else:
                print("Error is of unknown type (investigate further)")

            # Re-raise to see full traceback
            raise

    def test_document_loading_capability(self, real_rag_system):
        """Test if the system can load documents if they exist"""
        docs_path = "../docs"

        if os.path.exists(docs_path):
            try:
                # Try to load documents
                courses, chunks = real_rag_system.add_course_folder(docs_path)
                print(f"Successfully loaded {courses} courses with {chunks} chunks")

                # Test search after loading
                if courses > 0:
                    results = real_rag_system.vector_store.search("course")
                    if results.error:
                        print(f"Search after loading failed: {results.error}")
                    else:
                        print(f"Search after loading found {len(results.documents)} results")

            except Exception as e:
                print(f"Document loading failed: {e}")
                raise
        else:
            print("No docs folder found - skipping document loading test")

    def test_chromadb_connection(self, real_rag_system):
        """Test ChromaDB connection specifically"""
        try:
            # Try basic ChromaDB operations
            vector_store = real_rag_system.vector_store

            # Test collection access
            catalog_count = len(vector_store.course_catalog.get()['ids']) if vector_store.course_catalog.get()['ids'] else 0
            content_count = len(vector_store.course_content.get()['ids']) if vector_store.course_content.get()['ids'] else 0

            print(f"ChromaDB connection successful - Catalog: {catalog_count}, Content: {content_count}")

        except Exception as e:
            print(f"ChromaDB connection failed: {e}")
            # This might be our root cause
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print statements