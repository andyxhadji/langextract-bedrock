"""Bedrock provider implementation for LangExtract.

This module provides BedrockLanguageModel, which integrates AWS Bedrock's
Converse & invoke APIs with LangExtract for structured information extraction.

Supports two modes of operation:
1. Tool Use Mode (use_schema_constraints=True, fence_output=False):
   Uses Bedrock's Tool Use API to enforce structured JSON output.

2. Text Mode (fence_output=True):
   Model responds with ```json fenced output, parsed by langextract.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from typing import Any

import boto3
import langextract as lx

from langextract_bedrock.schema import BedrockToolUseSchema

AWS_DEFAULT_REGION = "us-east-1"


@lx.providers.registry.register(r"^bedrock/", priority=10)
class BedrockLanguageModel(lx.core.base_model.BaseLanguageModel):
    """LangExtract provider for Bedrock.

    This provider handles model IDs matching: ['^bedrock/']

    Example model IDs:
        - bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
        - bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
    """

    def __init__(
        self,
        model_id: str,
        api_method: str = "converse",
        max_workers: int = 1,
        **kwargs: Any,
    ):
        """Initialize the Bedrock provider.

        Args:
            model_id: The model identifier (with 'bedrock/' prefix).
            api_method: API method to use ('converse' or 'invoke').
            max_workers: Maximum number of workers for parallel inference.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        self.model_id = model_id.replace("bedrock/", "")
        self.process_prompt_fn = self._get_process_prompt_fn(api_method)
        self.max_workers = max_workers

        self.client = self._create_client()
        

    def _create_client(self) -> Any:
        """Create and return a Bedrock runtime client.

        Returns:
            boto3 Bedrock runtime client

        Raises:
            ValueError: If no valid AWS credentials are found.
        """
        has_bearer_token = "AWS_BEARER_TOKEN_BEDROCK" in os.environ
        has_aws_creds = (
            "AWS_ACCESS_KEY_ID" in os.environ
            and "AWS_SECRET_ACCESS_KEY" in os.environ
        )
        aws_profile = os.environ.get("AWS_PROFILE")
        has_default_region = "AWS_DEFAULT_REGION" in os.environ

        if not (has_bearer_token or has_aws_creds or aws_profile or has_default_region):
            raise ValueError(
                "AWS credentials required. Set one of: AWS_BEARER_TOKEN_BEDROCK, "
                "AWS_DEFAULT_REGION, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
                "or AWS_PROFILE"
            )

        region = os.environ.get("AWS_DEFAULT_REGION", AWS_DEFAULT_REGION)

        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            return session.client(service_name="bedrock-runtime", region_name=region)

        return boto3.client(service_name="bedrock-runtime", region_name=region)
    

    @classmethod
    def get_schema_class(cls) -> type[BedrockToolUseSchema]:
        """Return the schema class this provider supports.

        Returns:
            BedrockToolUseSchema class for use_schema_constraints=True
        """
        return BedrockToolUseSchema
    

    def _get_process_prompt_fn(self, api_method: str):
        """Get the appropriate prompt processing function.

        Args:
            api_method: 'converse' or 'invoke'

        Returns:
            The prompt processing function

        Raises:
            ValueError: If api_method is invalid
        """
        if api_method == "converse":
            return self._process_prompt_converse
        elif api_method == "invoke":
            return self._process_prompt_invoke
        else:
            raise ValueError(f"Invalid API method: {api_method}")
        

    def _build_inference_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build Bedrock inference configuration from kwargs.

        Args:
            kwargs: Keyword arguments that may contain inference parameters

        Returns:
            Dict with Bedrock-compatible inference config
        """
        config: dict[str, Any] = {}

        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            config["topP"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            config["maxTokens"] = kwargs["max_tokens"]
        if "max_tokens_to_sample" in kwargs:
            config["max_tokens_to_sample"] = kwargs["max_tokens_to_sample"]

        return config
    

    def _sanitize_tool_output(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Sanitize tool output to ensure langextract compatibility.

        Claude sometimes returns arrays or nested objects in attributes despite
        schema constraints. This method converts them to strings.

        Args:
            tool_input: Raw tool input from Bedrock response

        Returns:
            Sanitized dict with all attribute values as primitives
        """
        extractions = tool_input.get("extractions", [])
        sanitized_extractions = []

        for ext in extractions:
            sanitized_ext: dict[str, Any] = {}

            for key, value in ext.items():
                if key.endswith("_attributes") and isinstance(value, dict):
                    # Sanitize attribute values
                    sanitized_attrs: dict[str, Any] = {}
                    for attr_key, attr_value in value.items():
                        if isinstance(attr_value, list):
                            # Convert list to comma-separated string
                            sanitized_attrs[attr_key] = ", ".join(
                                str(v) for v in attr_value
                            )
                            logging.debug(
                                "[Bedrock] Converted array to string for '%s'",
                                attr_key,
                            )
                        elif isinstance(attr_value, dict):
                            # Convert dict to JSON string
                            sanitized_attrs[attr_key] = json.dumps(attr_value)
                            logging.debug(
                                "[Bedrock] Converted dict to string for '%s'",
                                attr_key,
                            )
                        else:
                            sanitized_attrs[attr_key] = attr_value
                    sanitized_ext[key] = sanitized_attrs

                elif isinstance(value, list):
                    # extraction_text should never be a list
                    sanitized_ext[key] = ", ".join(str(v) for v in value)
                    logging.debug(
                        "[Bedrock] Converted array to string for '%s'", key
                    )

                elif isinstance(value, dict) and not key.endswith("_attributes"):
                    # extraction_text should never be a dict
                    sanitized_ext[key] = json.dumps(value)
                    logging.debug(
                        "[Bedrock] Converted dict to string for '%s'", key
                    )

                else:
                    sanitized_ext[key] = value

            sanitized_extractions.append(sanitized_ext)

        return {"extractions": sanitized_extractions}
    

    def _extract_json_from_text(self, text: str) -> dict[str, Any] | None:
        """Try to extract JSON object from text that may contain prose.

        This is a fallback for when Claude responds with text instead of
        using the Tool Use API.

        Args:
            text: Text that may contain embedded JSON

        Returns:
            Extracted dict or None if no valid JSON found
        """
        import re

        # Try to find JSON block with extractions key
        patterns = [
            r"```json\s*(\{[\s\S]*?\})\s*```",  # ```json { } ```
            r"```\s*(\{[\s\S]*?\})\s*```",  # ``` { } ```
            r'(\{[^{}]*"extractions"[^{}]*\[[\s\S]*?\]\s*\})',  # inline JSON
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # Last resort: try to find any JSON object with extractions
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                potential_json = text[start : end + 1]
                parsed = json.loads(potential_json)
                if isinstance(parsed, dict) and "extractions" in parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

        return None
    

    def _process_prompt_converse(
        self,
        prompt: str,
        config: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: dict[str, Any] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> str:
        """Process a prompt using Bedrock's Converse API.

        Args:
            prompt: The prompt text
            config: Inference configuration
            tools: Optional list of tool specifications
            tool_executor: Optional dict mapping tool names to executor functions
            tool_choice: Tool choice configuration

        Returns:
            The model's response as a string
        """
        if tool_choice is None:
            tool_choice = {"auto": {}}

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": config,
        }

        if tools:
            kwargs["toolConfig"] = {"tools": tools, "toolChoice": tool_choice}

        response = self.client.converse(**kwargs)
        content = response.get("output", {}).get("message", {}).get("content", [])

        # Check if model used a tool
        tool_use_part = next(
            (
                p.get("toolUse")
                for p in content
                if isinstance(p, dict) and "toolUse" in p
            ),
            None,
        )

        # Handle tool execution (for custom tools with executors)
        if tool_use_part and tool_executor:
            return self._handle_tool_execution(
                tool_use_part, tool_executor, messages, config, tools
            )

        # Handle schema-based extraction (Tool Use without executor)
        if tool_use_part and not tool_executor:
            tool_input = tool_use_part.get("input") or {}
            logging.debug("[Bedrock] Tool input: %s", json.dumps(tool_input, indent=2))

            # Sanitize output to ensure langextract compatibility
            sanitized = self._sanitize_tool_output(tool_input)
            return json.dumps(sanitized)

        # Handle text response (no tool use)
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_content = part["text"]

                # If tools were provided but not used, try to extract JSON
                if tools and not tool_executor:
                    logging.warning(
                        "[Bedrock] Claude returned text instead of using tool. "
                        "Attempting to extract JSON from response."
                    )
                    extracted = self._extract_json_from_text(text_content)
                    if extracted:
                        sanitized = self._sanitize_tool_output(extracted)
                        return json.dumps(sanitized)
                    logging.error(
                        "[Bedrock] Could not extract JSON from: %s...",
                        text_content[:200],
                    )

                return text_content

            if isinstance(part, dict) and "json" in part:
                return json.dumps(part["json"])

        return ""


    def _handle_tool_execution(
        self,
        tool_use_part: dict[str, Any],
        tool_executor: dict[str, Any],
        messages: list[dict[str, Any]],
        config: dict[str, Any],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """Handle tool execution and follow-up conversation.

        Args:
            tool_use_part: The tool use request from Claude
            tool_executor: Dict mapping tool names to executor functions
            messages: The conversation messages
            config: Inference configuration
            tools: Tool specifications

        Returns:
            The final response after tool execution
        """
        tool_name = tool_use_part.get("toolName") or tool_use_part.get("name")
        tool_input = tool_use_part.get("input") or {}
        tool_use_id = tool_use_part.get("toolUseId") or tool_use_part.get(
            "id", "tool-1"
        )

        if tool_name in tool_executor:
            try:
                tool_result = tool_executor[tool_name](tool_input)
            except Exception as exc:
                tool_result = {"error": str(exc)}
        else:
            tool_result = {"error": f"No executor for tool '{tool_name}'"}

        followup_messages = [
            messages[0],
            {"role": "assistant", "content": [{"toolUse": tool_use_part}]},
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": tool_result}],
                        }
                    }
                ],
            },
        ]

        followup_kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": followup_messages,
            "inferenceConfig": config,
        }
        if tools:
            followup_kwargs["toolConfig"] = {"tools": tools}

        response = self.client.converse(**followup_kwargs)
        content = response.get("output", {}).get("message", {}).get("content", [])

        for part in content:
            if isinstance(part, dict) and "text" in part:
                return part["text"]
            if isinstance(part, dict) and "json" in part:
                return json.dumps(part["json"])

        return ""
    

    def _process_prompt_invoke(
        self,
        prompt: str,
        config: dict[str, Any],
        **_: Any,
    ) -> str:
        """Process a prompt using Bedrock's Invoke API (legacy).

        Args:
            prompt: The prompt text
            config: Inference configuration
            **_: Ignored additional arguments

        Returns:
            The model's response as a string
        """
        body = {"prompt": prompt}
        body.update(config)

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        return response.get("body").read()


    def infer(self, batch_prompts: list[str], **kwargs: Any):
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters including:
                - temperature: Sampling temperature (0.0 for deterministic)
                - top_p: Top-p sampling parameter
                - max_tokens: Maximum tokens to generate
                - schema: JSON schema for structured output (from use_schema_constraints)

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        config = self._build_inference_config(kwargs)

        # Check if schema was provided by langextract (use_schema_constraints=True)
        schema_dict = kwargs.get("schema")

        if schema_dict is not None:
            # Use Tool Use API for structured output
            tools = [
                {
                    "toolSpec": {
                        "name": "extract_structured_data",
                        "description": (
                            "MANDATORY: You MUST call this tool to return extraction results. "
                            "All attribute values must be simple strings or numbers. "
                            "For multiple values, use comma-separated strings."
                        ),
                        "inputSchema": {"json": schema_dict},
                    }
                }
            ]
            # Force tool usage
            tool_choice: dict[str, Any] = {"any": {}}
        else:
            # No schema - use legacy behavior
            tools = kwargs.get("tools")
            tool_choice = kwargs.get("tool_choice", {"auto": {}})

        tool_executor = kwargs.get("tool_executor")

        # Process prompts
        if len(batch_prompts) > 1 and self.max_workers > 1:
            yield from self._process_batch_parallel(
                batch_prompts, config, tools, tool_executor, tool_choice
            )
        else:
            yield from self._process_batch_sequential(
                batch_prompts, config, tools, tool_executor, tool_choice
            )


    def _process_batch_parallel(
        self,
        batch_prompts: list[str],
        config: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        tool_executor: dict[str, Any] | None,
        tool_choice: dict[str, Any],
    ):
        """Process prompts in parallel using ThreadPoolExecutor."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(batch_prompts))
        ) as executor:
            future_to_index = {
                executor.submit(
                    self.process_prompt_fn,
                    prompt,
                    config,
                    tools,
                    tool_executor,
                    tool_choice,
                ): i
                for i, prompt in enumerate(batch_prompts)
            }

            results: list[lx.inference.ScoredOutput | None] = [None] * len(
                batch_prompts
            )

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    output = future.result()
                    results[index] = lx.inference.ScoredOutput(score=1.0, output=output)
                except Exception as e:
                    raise RuntimeError(f"Parallel inference error: {str(e)}") from e

            for result in results:
                if result is None:
                    raise RuntimeError("Failed to process one or more prompts")
                yield [result]


    def _process_batch_sequential(
        self,
        batch_prompts: list[str],
        config: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        tool_executor: dict[str, Any] | None,
        tool_choice: dict[str, Any],
    ):
        """Process prompts sequentially."""
        for prompt in batch_prompts:
            output = self.process_prompt_fn(
                prompt,
                config,
                tools=tools,
                tool_executor=tool_executor,
                tool_choice=tool_choice,
            )
            yield [lx.inference.ScoredOutput(score=1.0, output=output)]
