# Tools constructed as Iterable[ChatCompletionToolParam]
import sys
import types

from openai.types.chat import ChatCompletionToolParam

from areal.experimental.openai.tool_call_parser import process_tool_calls

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "A powerful web search tool for accessing a vast range of external information beyond its training data. "
                "Use this tool when you need detailed information on highly specific, specialized, or niche topics, or when you need to verify information and fact-check claims by finding authoritative sources. "
                "It helps you answer complex questions that require deep knowledge or specific external data. The input should be a clear search query designed to find specific knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A precise search query for information retrieval.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "access",
            "description": (
                "Invokes the Jina AI Reader engine to intelligently access and parse a URL. "
                "This tool takes a webpage URL as input and returns its main article content in a clean Markdown format. "
                "Use this to perform a 'deep dive' on the most relevant link found via web_search to extract detailed evidence and data needed to answer a question. "
                "It automatically ignores advertisements and boilerplate code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the webpage to read.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


def test_process_tool_calls_qwen25_chat_completions():
    """
    Validate that process_tool_calls extracts tool calls from assistant text
    using the qwen25 parser and returns ChatCompletionMessageToolCall entries
    when use_responses=False.
    """
    text = (
        '<think>\nOkay, so the user is asking whether the director of "Scary Movie" and the director of "The Preacher\'s Wife" are from the same country. Let me think about how to approach this.\n\n'
        'First, I need to confirm the countries of these directors. I remember that "Scary Movie" is directed by Joe Anderson, and "The Preacher\'s Wife" is directed by David Fincher. Wait, but actually, "The Preacher\'s Wife" is directed by Christopher Nolan and David Fincher? No, I think I mixed up. Let me check my memory. \n\n'
        "Wait, no. The Preacher's Wife is a movie directed by Christopher Nolan. And David Fincher directed The Preacher. So the directors are different. Then the user is asking if they are both from the same country. The answer would be no, because the two directors are from different countries. \n\n"
        'But maybe I should verify this to be sure. Since I can use the web search function, I should use the "access" tool to get the URLs of the directors\' websites to confirm. So first, I\'ll search for "director of Scary Movie country" and "director of The Preacher\'s Wife country" to get precise data. Then, analyze the results to see if they have the same country affiliations.\n</think>\n\n'
        '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n'
        '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>'
    )

    tool_call_parser = "qwen25"
    reasoning_parser = "qwen3"
    finish_reason = "tool_calls"
    use_responses = False

    tool_calls, new_text, new_finish_reason = process_tool_calls(
        text=text,
        tools=tools,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        finish_reason=finish_reason,
        use_responses=use_responses,
    )

    # Assertions
    assert new_finish_reason == "tool_calls"
    assert tool_calls is not None, "Tool calls should be detected and returned"
    assert len(tool_calls) == 2, "Two tool calls should be parsed from the text"
    # Validate each parsed call is ChatCompletionMessageToolCall
    from openai.types.chat import ChatCompletionMessageToolCall

    assert isinstance(tool_calls[0], ChatCompletionMessageToolCall)
    assert isinstance(tool_calls[1], ChatCompletionMessageToolCall)
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search"
    assert tool_calls[1].function.name == "search"
    assert (
        tool_calls[0].function.arguments
        == '{"query": "director of Scary Movie country"}'
    )
    assert (
        tool_calls[1].function.arguments
        == '{"query": "director of The Preacher\'s Wife country"}'
    )
    # Ensure the returned text no longer contains raw <tool_call> blocks
    assert "<tool_call>" not in new_text


def test_process_tool_calls_qwen25_chat_completions_with_tool_call_in_thinking():
    """
    Validate that process_tool_calls extracts tool calls from assistant text
    using the qwen25 parser and returns ChatCompletionMessageToolCall entries
    when use_responses=False.
    """
    text = (
        '<think>\nOkay, so the user is asking whether the director of "Scary Movie" and the director of "The Preacher\'s Wife" are from the same country. Let me think about how to approach this.\n\n'
        'First, I need to confirm the countries of these directors. I remember that "Scary Movie" is directed by Joe Anderson, and "The Preacher\'s Wife" is directed by David Fincher. Wait, but actually, "The Preacher\'s Wife" is directed by Christopher Nolan and David Fincher? No, I think I mixed up. Let me check my memory. \n\n'
        'Wait, no. The Preacher\'s Wife is a movie directed by Christopher Nolan. And David Fincher directed The Preacher. <tool_call>\n{"name": "search", "arguments": {"query": "aaaa"}}\n</tool_call>\n\n So the directors are different. Then the user is asking if they are both from the same country. The answer would be no, because the two directors are from different countries. \n\n'
        'But maybe I should verify this to be sure. Since I can use the web search function, I should use the "access" tool to get the URLs of the directors\' websites to confirm. So first, I\'ll search for "director of Scary Movie country" and "director of The Preacher\'s Wife country" to get precise data. Then, analyze the results to see if they have the same country affiliations.\n</think>\n\n'
        '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n'
        '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>'
    )

    tool_call_parser = "qwen25"
    reasoning_parser = "qwen3"
    finish_reason = "tool_calls"
    use_responses = False

    tool_calls, new_text, new_finish_reason = process_tool_calls(
        text=text,
        tools=tools,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        finish_reason=finish_reason,
        use_responses=use_responses,
    )

    # Assertions
    assert new_finish_reason == "tool_calls"
    assert tool_calls is not None, "Tool calls should be detected and returned"
    assert len(tool_calls) == 2, "Two tool calls should be parsed from the text"
    # Validate each parsed call is ChatCompletionMessageToolCall
    from openai.types.chat import ChatCompletionMessageToolCall

    assert isinstance(tool_calls[0], ChatCompletionMessageToolCall)
    assert isinstance(tool_calls[1], ChatCompletionMessageToolCall)
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search"
    assert tool_calls[1].function.name == "search"
    assert (
        tool_calls[0].function.arguments
        == '{"query": "director of Scary Movie country"}'
    )
    assert (
        tool_calls[1].function.arguments
        == '{"query": "director of The Preacher\'s Wife country"}'
    )
    # Ensure the returned text no longer contains raw <tool_call> blocks
    assert "<tool_call>" in new_text


def test_process_tool_calls_falls_back_when_sglang_returns_empty(monkeypatch):
    """
    SGLang's parser can detect Qwen XML but fail to extract call info for some
    client-provided tool schemas. In that case the XML fallback must still run.
    """

    class EmptySGLangParser:
        def __init__(self, tools, tool_call_parser):
            pass

        def has_tool_call(self, content_text):
            return True

        def parse_non_stream(self, content_text):
            return content_text, []

    class SglFunction:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SglTool:
        def __init__(self, type, function):
            self.type = type
            self.function = function

    class ReasoningParser:
        def __init__(self, reasoning_parser):
            self.detector = types.SimpleNamespace(
                think_start_token="<think>",
                think_end_token="</think>",
            )

    protocol_module = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    protocol_module.Function = SglFunction
    protocol_module.Tool = SglTool
    function_parser_module = types.ModuleType(
        "sglang.srt.function_call.function_call_parser"
    )
    function_parser_module.FunctionCallParser = EmptySGLangParser
    reasoning_parser_module = types.ModuleType("sglang.srt.parser.reasoning_parser")
    reasoning_parser_module.ReasoningParser = ReasoningParser

    for module_name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
        "sglang.srt.function_call",
        "sglang.srt.parser",
    ]:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.openai.protocol", protocol_module
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.function_call.function_call_parser",
        function_parser_module,
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.parser.reasoning_parser", reasoning_parser_module
    )

    text = (
        '<tool_call>\n{"name": "search", "arguments": {"query": "current directory"}}\n'
        "</tool_call>"
    )

    tool_calls, new_text, new_finish_reason = process_tool_calls(
        text=text,
        tools=tools,
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        finish_reason="stop",
        use_responses=False,
    )

    assert new_finish_reason == "tool_calls"
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search"
    assert "<tool_call>" not in new_text


def test_process_tool_calls_falls_back_for_unclosed_thinking_block():
    text = (
        "<think>\n"
        "I should inspect the current directory.\n"
        '<tool_call>\n{"name": "search", "arguments": {"query": "current directory"}}\n'
        "</tool_call>"
    )

    tool_calls, new_text, new_finish_reason = process_tool_calls(
        text=text,
        tools=tools,
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        finish_reason="stop",
        use_responses=False,
    )

    assert new_finish_reason == "tool_calls"
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search"
    assert "<tool_call>" not in new_text
