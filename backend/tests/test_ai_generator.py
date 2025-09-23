import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class MockAnthropicResponse:
    """Mock response from Anthropic API"""

    def __init__(self, content_blocks, stop_reason="end_turn"):
        self.content = content_blocks
        self.stop_reason = stop_reason


class MockToolUseBlock:
    """Mock tool use content block"""

    def __init__(self, tool_name, tool_input, tool_id="test_id"):
        self.type = "tool_use"
        self.name = tool_name
        self.input = tool_input
        self.id = tool_id


class MockTextBlock:
    """Mock text content block"""

    def __init__(self, text):
        self.type = "text"
        self.text = text


class TestAIGenerator:
    """Tests for AI Generator tool integration"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        client = Mock()
        return client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AI Generator with mocked client"""
        with patch(
            "ai_generator.anthropic.Anthropic", return_value=mock_anthropic_client
        ):
            generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
            generator.client = mock_anthropic_client
            return generator

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        manager = Mock()
        manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }
        ]
        return manager

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test simple response generation without tools"""
        # Mock response without tool use
        mock_response = MockAnthropicResponse(
            [MockTextBlock("This is a simple response")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response("What is 2+2?")

        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args

        assert call_args[1]["model"] == "claude-sonnet-4-20250514"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert call_args[1]["messages"][0]["content"] == "What is 2+2?"
        assert "tools" not in call_args[1]

        assert result == "This is a simple response"

    def test_generate_response_with_tools_no_use(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test response generation with tools available but not used"""
        # Mock response without tool use
        mock_response = MockAnthropicResponse(
            [MockTextBlock("This is a general knowledge response")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response(
            "What is Python?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify API call included tools
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"] == {"type": "auto"}
        assert len(call_args[1]["tools"]) == 1

        assert result == "This is a general knowledge response"

    def test_generate_response_with_tool_use(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test response generation with tool use"""
        # Mock initial response with tool use
        initial_response = MockAnthropicResponse(
            [MockToolUseBlock("search_course_content", {"query": "Python basics"})],
            stop_reason="tool_use",
        )

        # Mock final response after tool execution
        final_response = MockAnthropicResponse(
            [
                MockTextBlock(
                    "Based on the search results, Python is a programming language..."
                )
            ]
        )

        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool execution
        mock_tool_manager.execute_tool.return_value = "Python course content found"

        result = ai_generator.generate_response(
            "What is Python?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2

        # Verify final response
        assert (
            result == "Based on the search results, Python is a programming language..."
        )

    def test_generate_response_with_conversation_history(
        self, ai_generator, mock_anthropic_client
    ):
        """Test response generation with conversation history"""
        mock_response = MockAnthropicResponse(
            [MockTextBlock("Continuing the conversation...")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: Hello\nAssistant: Hi there!"
        result = ai_generator.generate_response(
            "How are you?", conversation_history=history
        )

        # Verify system message includes history
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:\n" + history in system_content

        assert result == "Continuing the conversation..."

    def test_handle_tool_execution_multiple_tools(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test handling multiple tool calls in one response"""
        # Create initial response with multiple tool uses
        initial_response = MockAnthropicResponse(
            [
                MockToolUseBlock("search_course_content", {"query": "Python"}, "tool1"),
                MockToolUseBlock(
                    "search_course_content", {"query": "JavaScript"}, "tool2"
                ),
            ],
            stop_reason="tool_use",
        )

        # Mock final response
        final_response = MockAnthropicResponse(
            [MockTextBlock("Combined results from multiple searches")]
        )

        # Mock tool executions
        mock_tool_manager.execute_tool.side_effect = [
            "Python content",
            "JavaScript content",
        ]

        # Prepare base parameters
        base_params = {
            "messages": [{"role": "user", "content": "Compare Python and JavaScript"}],
            "system": "Test system message",
        }

        result = ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify tool calls
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("search_course_content",)
        assert calls[0][1] == {"query": "Python"}
        assert calls[1][0] == ("search_course_content",)
        assert calls[1][1] == {"query": "JavaScript"}

        assert result == "Combined results from multiple searches"

    def test_handle_tool_execution_error(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test handling tool execution errors"""
        # Mock initial response with tool use
        initial_response = MockAnthropicResponse(
            [MockToolUseBlock("search_course_content", {"query": "test"})],
            stop_reason="tool_use",
        )

        # Mock final response
        final_response = MockAnthropicResponse(
            [MockTextBlock("Error handled response")]
        )

        mock_anthropic_client.messages.create.return_value = final_response

        # Mock tool execution error
        mock_tool_manager.execute_tool.return_value = "Tool error: Database unavailable"

        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system",
        }

        result = ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Verify error was passed to final API call
        final_call_args = mock_anthropic_client.messages.create.call_args
        messages = final_call_args[1]["messages"]

        # Should have user message, assistant message, and tool result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert "Tool error: Database unavailable" in str(messages[2]["content"])

        assert result == "Error handled response"

    def test_system_prompt_formatting(self, ai_generator):
        """Test system prompt construction"""
        # Test without history
        generator = ai_generator

        # Access the static system prompt
        system_prompt = generator.SYSTEM_PROMPT

        assert "AI assistant specialized in course materials" in system_prompt
        assert "Search Tool Usage:" in system_prompt
        assert "One search per query maximum" in system_prompt

    def test_api_parameters_construction(self, ai_generator, mock_anthropic_client):
        """Test API parameter construction"""
        mock_response = MockAnthropicResponse([MockTextBlock("test")])
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response("test query")

        call_args = mock_anthropic_client.messages.create.call_args[1]

        # Verify required parameters
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "test query"

    def test_tool_choice_auto_setting(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test that tool_choice is set to auto when tools are provided"""
        mock_response = MockAnthropicResponse([MockTextBlock("test")])
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response(
            "test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["tool_choice"] == {"type": "auto"}

    def test_message_flow_in_tool_execution(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test proper message flow during tool execution"""
        # Mock tool use response
        initial_response = MockAnthropicResponse(
            [
                MockTextBlock("I'll search for that information."),
                MockToolUseBlock("search_course_content", {"query": "test"}),
            ],
            stop_reason="tool_use",
        )

        # Mock final response
        final_response = MockAnthropicResponse(
            [MockTextBlock("Based on the search results...")]
        )

        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]
        mock_tool_manager.execute_tool.return_value = "Search results found"

        base_params = {
            "messages": [{"role": "user", "content": "Original query"}],
            "system": "System prompt",
        }

        result = ai_generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Verify final API call structure
        final_call = mock_anthropic_client.messages.create.call_args_list[1]
        final_params = final_call[1]

        # Should have 3 messages: user, assistant (with tool use), user (with tool results)
        assert len(final_params["messages"]) == 3

        # Original user message
        assert final_params["messages"][0]["role"] == "user"
        assert final_params["messages"][0]["content"] == "Original query"

        # Assistant response with tool use
        assert final_params["messages"][1]["role"] == "assistant"
        assert final_params["messages"][1]["content"] == initial_response.content

        # Tool results
        assert final_params["messages"][2]["role"] == "user"
        tool_results = final_params["messages"][2]["content"]
        assert isinstance(tool_results, list)
        assert tool_results[0]["type"] == "tool_result"
        assert "Search results found" in tool_results[0]["content"]

        # Should not include tools in final call
        assert "tools" not in final_params

        assert result == "Based on the search results..."


class TestAIGeneratorIntegration:
    """Integration tests with real tool manager"""

    @pytest.fixture
    def real_tool_manager(self):
        """Create real tool manager with mocked vector store"""
        manager = ToolManager()
        mock_vector_store = Mock()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        return manager, mock_vector_store

    @pytest.fixture
    def ai_generator_real(self):
        """Create AI generator with real Anthropic client mock"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            client = Mock()
            mock_anthropic.return_value = client
            generator = AIGenerator("test_key", "claude-sonnet-4-20250514")
            return generator, client

    def test_end_to_end_tool_execution(self, ai_generator_real, real_tool_manager):
        """Test complete tool execution flow with real tool manager"""
        ai_generator, mock_client = ai_generator_real
        tool_manager, mock_vector_store = real_tool_manager

        # Setup search results
        mock_search_results = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = mock_search_results

        # Mock Claude responses
        tool_use_response = MockAnthropicResponse(
            [
                MockToolUseBlock(
                    "search_course_content", {"query": "Python programming"}
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockAnthropicResponse(
            [MockTextBlock("Python is a versatile programming language used for...")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Execute
        result = ai_generator.generate_response(
            "What is Python?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="Python programming", course_name=None, lesson_number=None
        )

        # Verify sources were tracked
        sources = tool_manager.get_last_sources()
        assert "Python Basics - Lesson 1" in sources

        assert result == "Python is a versatile programming language used for..."


if __name__ == "__main__":
    pytest.main([__file__])
