import os
import textwrap

import pytest

from .utils import aws_available


class DummyConverseClient:

    def converse(self, *, modelId, messages, inferenceConfig, **kwargs):  # noqa: N803
        # Return a simple text response; langextract should tolerate plain text
        prompt = messages[0]["content"][0]["text"]
        return {
            "output": {
                "message": {
                    "content": [{"text": f"Mock extraction for: {prompt[:40]}"}]
                }
            }
        }


def _ensure_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def test_lx_extract_with_examples_smoke(monkeypatch):
    if not aws_available():
        pytest.skip("AWS Bedrock environment not configured")

    import langextract as lx

    from langextract_bedrock import provider as _  # noqa: F401

    lx.providers.load_plugins_once()

    input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

    prompt = textwrap.dedent(
        """\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context."""
    )
    examples = [
        lx.data.ExampleData(
            text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"emotional_state": "wonder"},
                ),
                lx.data.Extraction(
                    extraction_class="emotion",
                    extraction_text="But soft!",
                    attributes={"feeling": "gentle awe"},
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="Juliet is the sun",
                    attributes={"type": "metaphor"},
                ),
            ],
        )
    ]

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        extraction_passes=1,
        max_workers=1,
    )

    assert result is not None


def test_lx_extract_multiple_workers(monkeypatch):
    if not aws_available():
        pytest.skip("AWS Bedrock environment not configured")

    import langextract as lx

    from langextract_bedrock import provider as _  # noqa: F401

    lx.providers.load_plugins_once()

    documents = [
        "Alice met Bob in Paris.",
        "John and Jane visited Boston.",
    ]

    prompts = [
        "Extract all person names mentioned in the text.",
        "Extract all location names mentioned in the text.",
    ]

    results = []
    for prompt in prompts:
        res = lx.extract(
            text_or_documents=documents,
            prompt_description=prompt,
            examples=[
                lx.data.ExampleData(
                    text="Romeo met Juliet in New York",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="character",
                            extraction_text="Romeo",
                            attributes={"emotional_state": "wonder"},
                        ),
                        lx.data.Extraction(
                            extraction_class="location",
                            extraction_text="New York",
                            attributes={"feeling": "gentle awe"},
                        ),
                    ],
                )
            ],
            model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            extraction_passes=1,
            max_workers=2,
        )
        results.append(res)

    assert all(r is not None for r in results)
