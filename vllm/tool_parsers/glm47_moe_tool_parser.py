# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM-4.7 Tool Call Parser.

GLM-4.7 uses a slightly different tool call format compared to GLM-4.5:
  - The function name may appear on the same line as ``<tool_call>`` without
    a newline separator before the first ``<arg_key>``.
  - Tool calls may have zero arguments
    (e.g. ``<tool_call>func</tool_call>``).

This parser overrides the parent regex patterns to handle both formats.
"""

import regex as re

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

logger = init_logger(__name__)


class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    supports_required_and_named = True

    def adjust_request(self, request):
        request = ToolParser.adjust_request(self, request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # GLM-4.7 format: <tool_call>func_name[<arg_key>...]*</tool_call>
        # The function name is the first token: it ends at whitespace or the
        # first tag ('<'), so it works whether the name is followed by a
        # newline, whitespace, or directly by an arg tag.  The argument section
        # is captured as-is and may be empty (zero-argument calls) or slightly
        # malformed (recovered by func_arg_regex), rather than failing the
        # whole tool call.
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*([^\s<]+)\s*(.*?)</tool_call>", re.DOTALL
        )
        # The opening <arg_key> is optional: GLM occasionally drops it and emits
        # ``key</arg_key><arg_value>value</arg_value>``, which would otherwise
        # lose the argument.  Keys never contain '<', so matching the key as
        # non-'<' keeps the recovery from swallowing a neighbouring tag.
        self.func_arg_regex = re.compile(
            r"(?:<arg_key>)?([^<]*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
