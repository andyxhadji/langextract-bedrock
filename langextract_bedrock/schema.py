"""Schema implementation for Bedrock Tool Use API.

This module provides BedrockToolUseSchema, which generates JSON schemas
compatible with AWS Bedrock's Converse API Tool Use feature for structured
output extraction.

The schema follows the same pattern as GeminiSchema: each extraction class
becomes a property name with its value being the extracted text.

References:
   -  https://github.com/google/langextract/blob/main/langextract/providers/schemas/gemini.py
   -  https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import langextract as lx
from langextract.core import data


class BedrockToolUseSchema(lx.core.schema.BaseSchema):
    """Schema implementation that uses Bedrock Tool Use API for structured output.

    This schema generates a JSON schema with dynamic property names based on
    extraction classes, following the same pattern as GeminiSchema. This ensures
    compatibility with langextract's resolver which expects:

        {"extractions": [{"class_name": "value", "class_name_attributes": {...}}]}

    NOT the generic format:

        {"extractions": [{"extraction_class": "X", "extraction_text": "Y"}]}
    """


    def __init__(self, json_schema: dict[str, Any]):
        """Initialize with a JSON schema.

        Args:
            json_schema: JSON Schema dict compatible with Tool Use inputSchema
        """
        self._json_schema = json_schema


    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> BedrockToolUseSchema:
        """Creates a BedrockToolUseSchema from example extractions.

        Builds a JSON schema with a top-level "extractions" array. Each
        element in that array is an object containing the extraction class name
        as a property (not as a value), following the same pattern as GeminiSchema.

        Args:
            examples_data: A sequence of ExampleData objects containing extraction
                classes and attributes.
            attribute_suffix: String appended to each class name to form the
                attributes field name (defaults to "_attributes").

        Returns:
            A BedrockToolUseSchema instance with the JSON schema.
        """
        # Track attribute types for each category (same as GeminiSchema)
        extraction_categories: dict[str, dict[str, set[type]]] = {}

        for example in examples_data:
            for extraction in example.extractions:
                category = extraction.extraction_class
                if category not in extraction_categories:
                    extraction_categories[category] = {}

                if extraction.attributes:
                    for attr_name, attr_value in extraction.attributes.items():
                        if attr_name not in extraction_categories[category]:
                            extraction_categories[category][attr_name] = set()
                        extraction_categories[category][attr_name].add(type(attr_value))

        # Build extraction item properties (same structure as GeminiSchema)
        extraction_properties: dict[str, dict[str, Any]] = {}

        for category, attrs in extraction_categories.items():
            # The category name is the key, value is the extracted text
            extraction_properties[category] = {
                "type": "string",
                "description": f"Extracted text for '{category}'. Must be a string.",
            }

            # Build attributes schema
            attributes_field = f"{category}{attribute_suffix}"
            attr_properties: dict[str, dict[str, Any]] = {}

            if not attrs:
                # Default property for categories without attributes
                attr_properties["_unused"] = {"type": "string"}
            else:
                for attr_name, attr_types in attrs.items():
                    # IMPORTANT: Always use string type for attributes
                    # LangExtract resolver expects primitive values (str, int, float) only
                    # Arrays and nested objects cause parsing errors
                    attr_properties[attr_name] = {
                        "type": "string",
                        "description": f"Attribute '{attr_name}' for {category}. Must be a string.",
                    }

            extraction_properties[attributes_field] = {
                "type": "object",
                "description": f"Metadata attributes for '{category}'. All values must be strings.",
                "properties": attr_properties,
            }

        # Build the extraction item schema
        extraction_schema = {
            "type": "object",
            "properties": extraction_properties,
            "description": (
                "A single extraction. Use the field name as the extraction class "
                "(e.g., 'numero_fallo': 'value'). Do NOT use 'extraction_class' or 'extraction_text'."
            ),
        }

        # Build the full schema with extractions wrapper
        json_schema = {
            "type": "object",
            "properties": {
                data.EXTRACTIONS_KEY: {
                    "type": "array",
                    "description": (
                        "List of extractions. Each item should have the extraction class "
                        "as a property name (e.g., {'numero_fallo': 'FALLO N° 123'})."
                    ),
                    "items": extraction_schema,
                }
            },
            "required": [data.EXTRACTIONS_KEY],
        }

        return cls(json_schema)


    def to_provider_config(self) -> dict[str, Any]:
        """Return the JSON schema for use in provider kwargs.

        Returns:
            Dict with 'schema' key containing the JSON schema
        """
        return {"schema": self._json_schema}


    @property
    def requires_raw_output(self) -> bool:
        """Tool Use returns structured JSON directly without fence markers.

        Returns:
            True - Tool Use outputs raw JSON without ```json fences
        """
        return True


    @property
    def schema_dict(self) -> dict[str, Any]:
        """Returns the schema dictionary."""
        return self._json_schema
