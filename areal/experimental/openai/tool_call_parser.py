from __future__ import annotations

import json
import os
import re
import traceback
import uuid
from typing import Any

from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from areal.utils import logging

logger = logging.getLogger("ToolCallParser")

_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_DEBUG_PREVIEW_CHARS = 500


def _debug_enabled() -> bool:
    return os.getenv("AREAL_OPENAI_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _preview(text: str, limit: int = _DEBUG_PREVIEW_CHARS) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _tool_names(tools: list[Any], use_responses: bool) -> set[str]:
    names = set()
    for tool in tools:
        if use_responses:
            name = tool.get("name") if isinstance(tool, dict) else None
        else:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.add(name)
    return names


def _dump_arguments(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _parse_qwen_xml_tool_calls(
    content_text: str, tools: list[Any], use_responses: bool
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
]:
    """Fallback parser for Qwen XML-style <tool_call> blocks."""
    matches = list(_QWEN_TOOL_CALL_RE.finditer(content_text))
    if not matches:
        return None, content_text

    valid_tool_names = _tool_names(tools, use_responses)
    if _debug_enabled():
        logger.info(
            f"Qwen XML fallback saw {len(matches)} tool_call block(s); "
            f"valid_tools={sorted(valid_tool_names)} use_responses={use_responses} "
            f"content_preview={_preview(content_text)}"
        )
    tool_calls: list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] = []

    for match in matches:
        payload = match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse Qwen tool call payload: {_preview(payload)}"
            )
            return None, content_text

        call_items = parsed if isinstance(parsed, list) else [parsed]
        for item in call_items:
            if not isinstance(item, dict):
                if _debug_enabled():
                    logger.warning(
                        f"Qwen XML fallback rejected non-object tool call item: "
                        f"{_preview(str(item))}"
                    )
                return None, content_text
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = item.get("name") or fn.get("name")
            arguments = item.get("arguments", item.get("parameters", fn.get("arguments")))
            if not name or (valid_tool_names and name not in valid_tool_names):
                if _debug_enabled():
                    logger.warning(
                        f"Qwen XML fallback rejected tool call name={name!r}; "
                        f"valid_tools={sorted(valid_tool_names)}"
                    )
                return None, content_text
            arguments_str = _dump_arguments(arguments)
            if use_responses:
                tool_calls.append(
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=f"fc-{uuid.uuid4().hex[:24]}",
                        call_id=f"call_{uuid.uuid4().hex[:24]}",
                        name=name,
                        arguments=arguments_str,
                        status="completed",
                    )
                )
            else:
                tool_calls.append(
                    ChatCompletionMessageFunctionToolCall(
                        type="function",
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=Function(name=name, arguments=arguments_str),
                    )
                )

    content_without_calls = _QWEN_TOOL_CALL_RE.sub("", content_text).strip()
    if _debug_enabled():
        logger.info(f"Qwen XML fallback parsed {len(tool_calls)} tool call(s)")
    return tool_calls or None, content_without_calls


def _detect_think_and_return_ori_think(
    text: str, think_start_token: str, think_end_token: str
) -> tuple[str, str]:
    """
    return think text(with <think> and </think>) and normal text
    """
    # This code is copies from sglang https://github.com/sgl-project/sglang/blob/cb30d056e3bc1b2f70fa7c00e0844cfe15716d65/python/sglang/srt/parser/reasoning_parser.py#L18
    in_reasoning = think_start_token in text

    if not in_reasoning:
        return "", text

    # The text is considered to be in a reasoning block.
    processed_text = text.replace(think_start_token, "")

    if think_end_token not in processed_text:
        # Assume reasoning was truncated before `</think>` token
        return think_start_token + processed_text, ""

    # Extract reasoning content
    splits = processed_text.split(think_end_token, maxsplit=1)
    reasoning_text = splits[0]
    normal_text = splits[1]

    return think_start_token + reasoning_text + think_end_token, normal_text


# Modified from sglang
def process_tool_calls(
    text: str,
    tools: list[Any],
    tool_call_parser: str,
    reasoning_parser: str,
    finish_reason: str,
    use_responses: bool = False,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
    str,
]:
    """Process tool calls in the response"""
    raw_tools = tools
    try:
        from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
        from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.parser.reasoning_parser import ReasoningParser
    except ModuleNotFoundError:
        if _debug_enabled():
            logger.warning(
                "SGLang parser is unavailable; using Qwen XML fallback only."
            )
        reasoning_text, content_text = _detect_think_and_return_ori_think(
            text,
            "<think>",
            "</think>",
        )
        tool_calls, content_text = _parse_qwen_xml_tool_calls(
            content_text, raw_tools, use_responses
        )
        if tool_calls:
            if finish_reason == "stop":
                finish_reason = "tool_calls"
            return tool_calls, reasoning_text + content_text, finish_reason
        return None, text, finish_reason

    if use_responses:
        tools = [
            SglTool(
                type=tool["type"],
                function=SglFunction(
                    name=tool.get("name"),
                    description=tool.get("description"),
                    parameters=tool.get("parameters"),
                ),
            )
            for tool in tools
        ]
    else:
        tools = [
            SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
            for tool in tools
        ]

    parser_p = FunctionCallParser(tools, tool_call_parser)
    reasoning_parser_p = ReasoningParser(reasoning_parser)

    reasoning_text, content_text = _detect_think_and_return_ori_think(
        text,
        reasoning_parser_p.detector.think_start_token,
        reasoning_parser_p.detector.think_end_token,
    )

    if parser_p.has_tool_call(content_text):
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        try:
            content_text, call_info_list = parser_p.parse_non_stream(content_text)

            if use_responses:
                tool_calls = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=f"fc-{uuid.uuid4().hex[:24]}",
                        call_id=f"call_{uuid.uuid4().hex[:24]}",
                        name=call_info.name,
                        arguments=call_info.parameters,
                        status="completed",
                    )
                    for call_info in call_info_list
                ]
            else:
                tool_calls = [
                    ChatCompletionMessageFunctionToolCall(
                        type="function",
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=Function(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                    for call_info in call_info_list
                ]

            if _debug_enabled():
                logger.info(
                    f"SGLang parser parsed {len(tool_calls)} tool call(s); "
                    f"use_responses={use_responses} "
                    f"remaining_has_xml={'<tool_call>' in content_text}"
                )
            if tool_calls and "<tool_call>" not in content_text:
                return tool_calls, reasoning_text + content_text, finish_reason

            if _debug_enabled():
                logger.warning(
                    f"SGLang parser result is incomplete; "
                    f"tool_calls={len(tool_calls)} "
                    f"remaining_has_xml={'<tool_call>' in content_text}. "
                    "Trying Qwen XML fallback."
                )
            fallback_tool_calls, fallback_content_text = _parse_qwen_xml_tool_calls(
                content_text, raw_tools, use_responses
            )
            if fallback_tool_calls:
                return (
                    fallback_tool_calls,
                    reasoning_text + fallback_content_text,
                    finish_reason,
                )
            return tool_calls or None, reasoning_text + content_text, finish_reason
        except Exception as e:
            logger.error(f"Tool call parsing error: {e}")
            traceback.print_exc()
            tool_calls, content_text = _parse_qwen_xml_tool_calls(
                content_text, raw_tools, use_responses
            )
            if tool_calls:
                return tool_calls, reasoning_text + content_text, finish_reason
            # Return error but don't fail the whole request
            return None, text, finish_reason

    tool_calls, content_text = _parse_qwen_xml_tool_calls(
        content_text, raw_tools, use_responses
    )
    if tool_calls:
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        return tool_calls, reasoning_text + content_text, finish_reason

    return None, text, finish_reason
