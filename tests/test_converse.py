import json
import os

import pytest


class DummyConverseClientNoTools:

    def converse(self, *, modelId, messages, inferenceConfig, **kwargs):  # noqa: N803
        # Return a simple text response echoing part of the prompt
        prompt = messages[0]["content"][0]["text"]
        return {"output": {"message": {"content": [{"text": f"Echo: {prompt[:30]}"}]}}}


class DummyConverseClientTools:

    def __init__(self):
        self._seen_tool_result = False

    def converse(self, *, modelId, messages, inferenceConfig, **kwargs):  # noqa: N803
        # First call: instruct a tool use
        if not self._seen_tool_result:
            self._seen_tool_result = any(
                isinstance(part, dict) and "toolResult" in part
                for m in messages
                for part in m.get("content", [])
            )
            if not self._seen_tool_result:
                return {
                    "output": {
                        "message": {
                            "content": [
                                {
                                    "toolUse": {
                                        "toolName": "calculate_sum",
                                        "toolUseId": "t1",
                                        "input": {"a": 1, "b": 2},
                                    }
                                }
                            ]
                        }
                    }
                }

        # Follow-up call after tool result: return JSON content
        return {
            "output": {"message": {"content": [{"json": {"result": "ok", "sum": 3}}]}}
        }


from .utils import aws_available


def test_converse_without_tools(monkeypatch):
    if not aws_available():
        pytest.skip("AWS Bedrock environment not configured")

    from langextract_bedrock.provider import BedrockLanguageModel

    provider = BedrockLanguageModel(
        model_id=os.environ.get("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0")
    )

    prompts = ["Hello there", "What is the capital of France?"]
    results = list(provider.infer(prompts, temperature=0.0, top_p=1.0, max_tokens=128))

    assert len(results) == 2
    assert all(isinstance(batch, list) and len(batch) == 1 for batch in results)
    out0 = results[0][0].output
    assert "Paris" in results[1][0].output
    assert isinstance(out0, str) and len(out0) > 0


def test_converse_with_tools(monkeypatch):
    if not aws_available():
        pytest.skip("AWS Bedrock environment not configured")

    from langextract_bedrock.provider import BedrockLanguageModel

    tools = [
        {
            "toolSpec": {
                "name": "calculate_sum",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    }
                },
            }
        }
    ]

    def exec_calculate_sum(inp):
        return {"sum": inp.get("a", 0) + inp.get("b", 0)}

    tool_executor = {"calculate_sum": exec_calculate_sum}

    provider = BedrockLanguageModel(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    results = list(
        provider.infer(
            ["What is the sum of 1 and 2?"],
            tools=tools,
            tool_executor=tool_executor,
            tool_choice={"tool": {"name": "calculate_sum"}},
        )
    )

    assert len(results) == 1 and len(results[0]) == 1
    output = results[0][0].output
    assert "3" in output
