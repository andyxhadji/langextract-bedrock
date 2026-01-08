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


class DummyConverseClientSchemaExtraction:
    """Mock client for schema-based extraction (tool use without executor)."""

    def converse(self, *, modelId, messages, inferenceConfig, **kwargs):  # noqa: N803
        # Return toolUse with structured extraction data (schema-based extraction)
        # This simulates what happens when use_schema_constraints=True
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolName": "extract_structured_data",
                                "toolUseId": "tool-1",
                                "input": {
                                    "extractions": [
                                        {
                                            "extraction_class": "character",
                                            "extraction_text": "ROMEO",
                                            "attributes": {"family": "Montague"},
                                        },
                                        {
                                            "extraction_class": "character",
                                            "extraction_text": "JULIET",
                                            "attributes": {"family": "Capulet"},
                                        },
                                    ]
                                },
                            }
                        }
                    ]
                }
            }
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


def test_converse_schema_extraction_no_executor(monkeypatch):
    """Test schema-based extraction: tool use without tool_executor.
    
    This validates the fix for the bug where toolUse.input wasn't extracted
    when using schema constraints (use_schema_constraints=True).
    """
    from langextract_bedrock.provider import BedrockLanguageModel

    # Mock the Bedrock client to return toolUse with structured data
    mock_client = DummyConverseClientSchemaExtraction()

    provider = BedrockLanguageModel(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    provider.client = mock_client

    # Schema-based extraction: schema provided but no tool_executor
    schema_dict = {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "extraction_class": {"type": "string"},
                        "extraction_text": {"type": "string"},
                        "attributes": {"type": "object"},
                    },
                },
            }
        },
        "required": ["extractions"],
    }

    tools = [
        {
            "toolSpec": {
                "name": "extract_structured_data",
                "inputSchema": {"json": schema_dict},
            }
        }
    ]

    results = list(
        provider.infer(
            ["Extract characters from: ROMEO and JULIET met at the ball."],
            schema=schema_dict,
            tools=tools,
            tool_executor=None,  # No executor - this is the key scenario
            tool_choice={"any": {}},
            temperature=0.0,
        )
    )

    # Verify we got a result
    assert len(results) == 1 and len(results[0]) == 1
    output = results[0][0].output

    # The output should be JSON string containing the extractions
    assert isinstance(output, str)
    parsed = json.loads(output)

    # Verify the structure matches what was in toolUse.input
    assert "extractions" in parsed
    assert len(parsed["extractions"]) == 2
    assert parsed["extractions"][0]["extraction_class"] == "character"
    assert parsed["extractions"][0]["extraction_text"] == "ROMEO"
    assert parsed["extractions"][1]["extraction_text"] == "JULIET"
