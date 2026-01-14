"""LangExtract provider plugin for Bedrock."""

from langextract_bedrock.provider import BedrockLanguageModel
from langextract_bedrock.schema import BedrockToolUseSchema

__all__ = ["BedrockLanguageModel", "BedrockToolUseSchema"]
__version__ = "0.1.5"
