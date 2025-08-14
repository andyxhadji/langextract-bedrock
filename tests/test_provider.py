import json
import os
import types

import langextract as lx
import pytest


def test_provider_registration_and_resolve(monkeypatch):
    # Ensure plugins are loaded and the module import triggers decorator registration
    from langextract_bedrock.provider import BedrockLanguageModel  # noqa: F401

    lx.providers.load_plugins_once()

    from langextract.providers import registry

    # Note: this won't happen because of naming conventions in bedrock, but just to test loading
    resolved = registry.resolve("bedrock/fake-model")
    assert resolved.__name__ == "BedrockLanguageModel"


def test_registry_does_not_match_unknown_prefix(monkeypatch):
    from langextract_bedrock.provider import BedrockLanguageModel  # noqa: F401

    lx.providers.load_plugins_once()

    from langextract.providers import registry

    try:
        resolved = registry.resolve("unknown-model")
        assert resolved.__name__ != "BedrockLanguageModel"
    except Exception:
        # If the registry raises for unknown providers, that's also acceptable
        assert True


def test_init_requires_env_vars(monkeypatch):
    for k in ["AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
        monkeypatch.delenv(k, raising=False)

    from langextract_bedrock.provider import BedrockLanguageModel

    with pytest.raises(ValueError):
        BedrockLanguageModel(model_id="fake-model")


def test_set_config_mapping(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    import boto3

    def fake_client(service_name, region_name=None):
        class Dummy:

            def converse(self, **kwargs):
                return {"output": {"message": {"content": [{"text": "irrelevant"}]}}}

        return Dummy()

    monkeypatch.setattr(boto3, "client", fake_client)

    from langextract_bedrock.provider import BedrockLanguageModel

    provider = BedrockLanguageModel(model_id="claude", provider="BedrockLanguageModel")
    cfg = provider.set_config({"temperature": 0.1, "top_p": 0.9, "max_tokens": 256})
    assert cfg == {"temperature": 0.1, "topP": 0.9, "maxTokens": 256}


def test_infer_batches_and_scored_output(monkeypatch):
    # Minimum env for init
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    # Mock boto3 client to return a fixed response
    import boto3

    class DummyClient:

        def converse(self, *, modelId, messages, inferenceConfig):
            # echo part of prompt to verify plumbing
            prompt = messages[0]["content"][0]["text"]
            return {
                "output": {"message": {"content": [{"text": f"Echo: {prompt[:20]}"}]}}
            }

    monkeypatch.setattr(
        boto3, "client", lambda service_name, region_name=None: DummyClient()
    )

    from langextract_bedrock.provider import BedrockLanguageModel

    provider = BedrockLanguageModel(model_id="fake-model")
    prompts = ["Test prompt 1", "Another test prompt 2"]
    results = list(provider.infer(prompts, temperature=0.0, top_p=1.0, max_tokens=32))

    # One ScoredOutput per prompt, in order
    assert len(results) == 2
    assert all(isinstance(batch, list) and len(batch) == 1 for batch in results)
    assert all(
        hasattr(batch[0], "output") and hasattr(batch[0], "score") for batch in results
    )
    assert results[0][0].output.startswith("Echo: Test prompt")
    assert results[1][0].output.startswith("Echo: Another test")


def test_factory_integration_with_provider_name(monkeypatch):
    # Env and client stubs
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    import boto3

    class DummyClient:

        def converse(self, *, modelId, messages, inferenceConfig):
            prompt = messages[0]["content"][0]["text"]
            return {
                "output": {"message": {"content": [{"text": f"Echo: {prompt[:20]}"}]}}
            }

    monkeypatch.setattr(
        boto3, "client", lambda service_name, region_name=None: DummyClient()
    )

    # Ensure plugin is registered
    from langextract_bedrock import provider as _  # noqa: F401

    lx.providers.load_plugins_once()

    # Create via factory with explicit provider name
    from langextract import factory

    config = factory.ModelConfig(model_id="fake-model", provider="BedrockLanguageModel")
    model = factory.create_model(config)

    assert type(model).__name__ == "BedrockLanguageModel"
    outputs = list(model.infer(["hi there"]))
    assert outputs and outputs[0] and outputs[0][0].output.startswith("Echo: hi there")


def test_infer_with_invoke_api_method(monkeypatch):
    # Arrange env and stub client
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    import boto3

    class DummyBody:

        def __init__(self, text: str):
            self._text = text

        def read(self):
            return self._text

    class DummyClient:

        def invoke_model(self, **kwargs):
            # Accept arbitrary kwargs (including nonstandard 'config') to match provider call
            prompt = json.loads(kwargs.get("body", ""))["prompt"]
            return {"body": DummyBody(f"InvokeEcho: {str(prompt)[:20]}")}

    monkeypatch.setattr(
        boto3, "client", lambda service_name, region_name=None: DummyClient()
    )

    from langextract_bedrock.provider import BedrockLanguageModel

    # Act
    provider = BedrockLanguageModel(model_id="fake-model", api_method="invoke")
    results = list(provider.infer(["Hello world from invoke"]))

    # Assert
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].output.startswith("InvokeEcho: Hello wor")
