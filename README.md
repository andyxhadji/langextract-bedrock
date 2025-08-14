# LangExtract AWS Bedrock Provider

A provider plugin for LangExtract that supports AWS Bedrock models.

## Installation

```bash
pip install langextract-bedrock
```

## Supported Model IDs

You can use any AWS Bedrock model by specifying it's ARN, for example:

```python
config = factory.ModelConfig(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", provider="BedrockLanguageModel")
model = factory.create_model(config)
```

## Environment Variables

Set any of the following environment variables:

For credentials:

- `AWS_PROFILE`
- `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`

For settings:

- `AWS_DEFAULT_REGION` (defaults to us-east-1)

## Usage
Use with `lx.extract` by pre-pending the model ARN with `bedrock/`:

```python
import langextract as lx

result = lx.extract(
    text="Your document here",
    model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`, `pytest -v tests/` (run pytest with AWS creds for AWS Bedrock inference tests)
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
