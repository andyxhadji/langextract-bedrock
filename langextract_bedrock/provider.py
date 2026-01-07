"""Provider implementation for Bedrock."""

import concurrent.futures
import json
import os

import boto3
import langextract as lx

AWS_DEFAULT_REGION = "us-east-1"

class BedrockToolUseSchema(lx.core.schema.BaseSchema):
    """Schema implementation that uses Bedrock Tool Use API for structured output.
    
    This schema leverages Bedrock's Converse API Tool Use feature to enforce
    JSON structure.
    
    Reference: https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/
    """
    
    def __init__(self, json_schema: dict):
        """Initialize with a JSON schema.
        
        Args:
            json_schema: JSON Schema dict compatible with Tool Use inputSchema
        """
        self._json_schema = json_schema
    
    @classmethod
    def from_examples(
        cls,
        examples_data,
        attribute_suffix: str = "_attrs",
    ):
        """Build schema from examples by analyzing extraction classes.
        
        Args:
            examples_data: Sequence of ExampleData objects
            attribute_suffix: Suffix for attribute fields (unused, for compatibility)
            
        Returns:
            BedrockToolUseSchema instance
        """
        # Extract unique extraction classes from examples
        extraction_classes = set()
        for example in examples_data:
            for extraction in example.extractions:
                extraction_classes.add(extraction.extraction_class)
        
        # Build JSON Schema following langextract's expected structure
        json_schema = {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "description": (
                        "List of extracted entities. Create one object per entity - "
                        "do NOT use arrays in extraction_text."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "extraction_class": {
                                "type": "string",
                                "description": (
                                    f"Type of extraction. Must be one of: "
                                    f"{', '.join(sorted(extraction_classes))}"
                                ),
                                "enum": sorted(list(extraction_classes))
                            },
                            "extraction_text": {
                                "type": "string",
                                "description": (
                                    "The literal text from the document. "
                                    "MUST be a string, NOT an array. "
                                    "For multiple items, create separate objects."
                                )
                            },
                            "attributes": {
                                "type": "object",
                                "description": (
                                    "Additional metadata. CRITICAL: All values MUST be "
                                    "primitive types (string or number ONLY). "
                                    "Arrays and nested objects are STRICTLY FORBIDDEN. "
                                    "For multiple values, use comma-separated strings."
                                ),
                                "additionalProperties": {
                                    "type": ["string", "number"]
                                }
                            }
                        },
                        "required": ["extraction_class", "extraction_text"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["extractions"],
            "additionalProperties": False
        }
        
        return cls(json_schema)
    
    def to_provider_config(self) -> dict:
        """Return the JSON schema for use in provider kwargs.
        
        Returns:
            Dict with 'schema' key containing the JSON schema
        """
        return {"schema": self._json_schema}
    
    @property
    def requires_raw_output(self) -> bool:
        """Tool Use returns structured JSON directly without fence markers.
        
        Returns:
            True - Tool Use outputs raw JSON
        """
        return True


@lx.providers.registry.register(r"^bedrock/", priority=10)
class BedrockLanguageModel(lx.core.base_model.BaseLanguageModel):
    """LangExtract provider for Bedrock.

    This provider handles model IDs matching: ['^bedrock/']
    """

    def __init__(
        self,
        model_id: str,
        api_method: str = "converse",
        max_workers: int = 1,
        **kwargs,
    ):
        """Initialize the Bedrock provider.

        Args:
            model_id: The model identifier.
            api_key: API key for authentication.
            max_workers: The maximum number of workers to use for parallel inference.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        self.model_id = model_id.replace("bedrock/", "")
        self.process_prompt_fn = self.get_process_prompt_fn(api_method)
        self.max_workers = max_workers

        has_bearer_token = "AWS_BEARER_TOKEN_BEDROCK" in os.environ
        has_aws_creds = (
            "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ
        )
        aws_profile = os.environ.get("AWS_PROFILE", False)
        has_default_region = "AWS_DEFAULT_REGION" in os.environ

        if not (has_bearer_token or has_aws_creds or aws_profile or has_default_region):
            raise ValueError(
                "Either AWS_BEARER_TOKEN_BEDROCK, AWS_DEFAULT_REGION,"
                " AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE must be set"
            )

        # Set region, defaulting to us-east-1 if not specified
        if "AWS_DEFAULT_REGION" in os.environ:
            region = os.environ["AWS_DEFAULT_REGION"]
        else:
            region = AWS_DEFAULT_REGION

        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            self.client = session.client(
                service_name="bedrock-runtime", region_name=region
            )
        else:
            self.client = boto3.client(
                service_name="bedrock-runtime", region_name=region
            )
    
    @classmethod
    def get_schema_class(cls):
        """Return the schema class this provider supports.
        
        Returns BedrockToolUseSchema to enable use_schema_constraints=True.
        
        Returns:
            BedrockToolUseSchema class
        """
        return BedrockToolUseSchema

    def get_process_prompt_fn(self, api_method):
        if api_method == "converse":
            return self._process_prompt_converse
        elif api_method == "invoke":
            return self._process_prompt_invoke
        else:
            raise ValueError(f"Invalid API method: {api_method}")

    def set_config(self, kwargs):
        config = {}
        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            config["topP"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            config["maxTokens"] = kwargs["max_tokens"]
        if "max_tokens_to_sample" in kwargs:
            config["max_tokens_to_sample"] = kwargs["max_tokens_to_sample"]
        return config

    def _process_prompt_converse(
        self,
        prompt,
        config,
        tools=None,
        tool_executor=None,
        tool_choice={"auto": {}},
    ):
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": config,
        }
        if tools:
            kwargs["toolConfig"] = {"tools": tools, "toolChoice": tool_choice}

        response = self.client.converse(**kwargs)
        content = response.get("output", {}).get("message", {}).get("content", [])

        # If the model asks to use a tool, execute and send the result back
        tool_use_part = next(
            (
                p.get("toolUse")
                for p in content
                if isinstance(p, dict) and "toolUse" in p
            ),
            None,
        )
        if tool_use_part and tool_executor:
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

            followup_kwargs = {
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

    def _process_prompt_invoke(self, prompt, config, **_):
        body = {
            "prompt": prompt,
        }
        body.update(config)
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        return response.get("body").read()

    def infer(self, batch_prompts, **kwargs):
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        config = self.set_config(kwargs)

        # Check if schema was provided by langextract (when use_schema_constraints=True)
        schema_dict = kwargs.get("schema", None)
        
        if schema_dict is not None:
            # Schema provided - use Tool Use API to enforce structure
            # This is the AWS-recommended approach for reliable structured output
            tools = [{
                "toolSpec": {
                    "name": "extract_structured_data",
                    "description": (
                        "Extract structured information from text according to schema. "
                        "You MUST use this tool to return extraction results. "
                        "CRITICAL: In the 'attributes' object, all values must be "
                        "simple strings or numbers - never use arrays or nested objects. "
                        "For multiple values, use comma-separated strings like 'value1, value2'."
                    ),
                    "inputSchema": {
                        "json": schema_dict
                    }
                }
            }]
            # Force tool usage to ensure structured output
            tool_choice = {"any": {}}
        else:
            # No schema - use legacy behavior (manual tools from kwargs)
            tools = kwargs.get("tools", None)
            tool_choice = kwargs.get("tool_choice", {"auto": {}})
        
        tool_executor = kwargs.get("tool_executor", None)

        if len(batch_prompts) > 1 and self.max_workers > 1:
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

                results = [None] * len(batch_prompts)
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        output = future.result()
                        results[index] = lx.inference.ScoredOutput(
                            score=1.0, output=output
                        )
                    except Exception as e:
                        raise RuntimeError(f"Parallel inference error: {str(e)}") from e

                for result in results:
                    if result is None:
                        raise RuntimeError("Failed to process one or more prompts")
                    yield [result]
        else:
            for prompt in batch_prompts:
                output = self.process_prompt_fn(
                    prompt,
                    config,
                    tools=tools,
                    tool_executor=tool_executor,
                    tool_choice=tool_choice,
                )
                yield [lx.inference.ScoredOutput(score=1.0, output=output)]
