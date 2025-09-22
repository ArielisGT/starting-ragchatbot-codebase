import pytest
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem


class TestRealAPIIntegration:
    """Test the actual system with real API key to verify it works"""

    @pytest.fixture
    def real_rag_system(self):
        """Create RAG system with real configuration"""
        return RAGSystem(config)

    def test_real_system_query(self, real_rag_system):
        """Test a real query with the actual system"""
        try:
            # Test a course-related question that should trigger search
            response, sources = real_rag_system.query("What is MCP in the context of AI applications?")

            print(f"Response: {response[:200]}...")
            print(f"Sources: {sources}")

            assert isinstance(response, str)
            assert len(response) > 0
            assert isinstance(sources, list)

            # Should have found sources since we know MCP course exists
            if len(sources) > 0:
                print("SUCCESS: System found relevant sources")
            else:
                print("INFO: No sources found (Claude may have answered from general knowledge)")

        except Exception as e:
            print(f"FAILURE: Real system query failed: {e}")
            raise

    def test_real_system_general_query(self, real_rag_system):
        """Test a general knowledge query (should not use search)"""
        try:
            response, sources = real_rag_system.query("What is 2+2?")

            print(f"General response: {response}")
            print(f"General sources: {sources}")

            assert isinstance(response, str)
            assert len(response) > 0
            assert isinstance(sources, list)

            # General knowledge shouldn't use sources
            assert len(sources) == 0

        except Exception as e:
            print(f"FAILURE: General query failed: {e}")
            raise

    def test_real_system_course_specific_query(self, real_rag_system):
        """Test a course-specific query"""
        try:
            response, sources = real_rag_system.query("How do you use ChromaDB for vector search?")

            print(f"Course-specific response: {response[:200]}...")
            print(f"Course-specific sources: {sources}")

            assert isinstance(response, str)
            assert len(response) > 0
            assert isinstance(sources, list)

            # This should find sources about ChromaDB
            print(f"Found {len(sources)} sources")

        except Exception as e:
            print(f"FAILURE: Course-specific query failed: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print statements