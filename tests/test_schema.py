"""
Test script for verifying schema constraints support in langextract-bedrock.

Run: python -m pytest tests/test_schema.py -v
Or:  python tests/test_schema.py

Requirements:
- AWS credentials configured (for integration tests)
- langextract installed
- langextract-bedrock installed in editable mode (pip install -e .)

Tests are organized in two categories:
1. Unit tests (no AWS required): Test schema generation and structure
2. Integration tests (AWS required): Test actual extraction with Bedrock
"""

import json
import sys
import textwrap

import langextract as lx
from dotenv import load_dotenv

load_dotenv()


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# UNIT TESTS (No AWS Required)
# =============================================================================


def test_1_schema_class_exists():
    """Test 1: Verify BedrockToolUseSchema exists and is correct type."""
    print_section("TEST 1: Verify schema class exists")

    # Import from the new schema.py module
    from langextract_bedrock.schema import BedrockToolUseSchema

    # Verify it's a subclass of BaseSchema
    assert issubclass(
        BedrockToolUseSchema, lx.schema.BaseSchema
    ), "BedrockToolUseSchema is not a BaseSchema subclass"

    print("✓ BedrockToolUseSchema exists and extends BaseSchema")

    # Also verify it can be imported from __init__.py
    from langextract_bedrock import BedrockToolUseSchema as SchemaFromInit

    assert (
        SchemaFromInit is BedrockToolUseSchema
    ), "Import from __init__.py doesn't match schema.py"

    print("✓ BedrockToolUseSchema correctly exported from package")
    print("\n✅ TEST 1 PASSED")


def test_2_get_schema_class():
    """Test 2: Verify get_schema_class() returns BedrockToolUseSchema."""
    print_section("TEST 2: Verify get_schema_class() method")

    from langextract_bedrock.provider import BedrockLanguageModel
    from langextract_bedrock.schema import BedrockToolUseSchema

    # Check class method
    schema_class = BedrockLanguageModel.get_schema_class()

    assert schema_class is not None, "get_schema_class() returns None"
    assert (
        schema_class == BedrockToolUseSchema
    ), f"get_schema_class() returns {schema_class}, expected BedrockToolUseSchema"

    print("✓ get_schema_class() returns BedrockToolUseSchema")
    print("\n✅ TEST 2 PASSED")


def test_3_schema_generation_dynamic_properties():
    """Test 3: Verify schema generates dynamic properties (like GeminiSchema)."""
    print_section("TEST 3: Schema generation with dynamic properties")

    from langextract_bedrock.schema import BedrockToolUseSchema

    # Create test examples
    examples = [
        lx.data.ExampleData(
            text="ROMEO: But soft, what light through yonder window breaks?",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"family": "Montague"},
                ),
                lx.data.Extraction(
                    extraction_class="dialogue",
                    extraction_text="But soft, what light through yonder window breaks?",
                    attributes={"tone": "wonder"},
                ),
                lx.data.Extraction(
                    extraction_class="emotion",
                    extraction_text="soft",
                    attributes={"intensity": "gentle"},
                ),
            ],
        )
    ]

    # Generate schema
    schema_instance = BedrockToolUseSchema.from_examples(examples)

    # Verify it's an instance of BedrockToolUseSchema
    assert isinstance(
        schema_instance, BedrockToolUseSchema
    ), f"from_examples() returned {type(schema_instance)}"

    print("✓ from_examples() returns BedrockToolUseSchema instance")

    # Check to_provider_config() returns dict with schema
    config = schema_instance.to_provider_config()

    assert isinstance(
        config, dict
    ), f"to_provider_config() returns {type(config)}, expected dict"
    assert (
        "schema" in config
    ), f"to_provider_config() missing 'schema' key. Keys: {config.keys()}"

    print("✓ to_provider_config() returns dict with 'schema' key")

    # Verify schema structure
    schema_dict = config["schema"]

    assert (
        "properties" in schema_dict and "extractions" in schema_dict["properties"]
    ), f"Schema missing expected structure: {json.dumps(schema_dict, indent=2)}"

    print("✓ Schema has correct top-level structure")

    # KEY TEST: Verify dynamic properties (NOT extraction_class/extraction_text)
    items = schema_dict["properties"]["extractions"]["items"]
    item_properties = items.get("properties", {})

    # Check that category names are used as property keys
    expected_categories = {"character", "dialogue", "emotion"}
    actual_categories = {
        k for k in item_properties.keys() if not k.endswith("_attributes")
    }

    assert expected_categories == actual_categories, (
        f"Schema should use category names as property keys. "
        f"Expected: {expected_categories}, Actual: {actual_categories}"
    )

    print(f"✓ Schema uses dynamic property names: {actual_categories}")

    # Verify that extraction_class/extraction_text are NOT in schema
    assert "extraction_class" not in item_properties, (
        "Schema should NOT contain extraction_class"
    )
    assert "extraction_text" not in item_properties, (
        "Schema should NOT contain extraction_text"
    )

    print("✓ Schema correctly avoids generic extraction_class/extraction_text")

    # Verify attributes fields exist
    expected_attr_fields = {
        "character_attributes",
        "dialogue_attributes",
        "emotion_attributes",
    }
    actual_attr_fields = {
        k for k in item_properties.keys() if k.endswith("_attributes")
    }

    assert expected_attr_fields == actual_attr_fields, (
        f"Attribute fields mismatch. Expected: {expected_attr_fields}, "
        f"Actual: {actual_attr_fields}"
    )

    print(f"✓ Attribute fields correct: {actual_attr_fields}")

    # Check requires_raw_output
    assert (
        schema_instance.requires_raw_output
    ), "requires_raw_output should be True for Tool Use"

    print("✓ requires_raw_output is True")

    # Show generated schema
    print("\n📋 Generated schema (truncated):")
    print(json.dumps(schema_dict, indent=2, ensure_ascii=False)[:800] + "...")

    print("\n✅ TEST 3 PASSED")


def test_4_schema_attribute_types():
    """Test 4: Verify all attribute types are strings (not arrays)."""
    print_section("TEST 4: Attribute types are all strings")

    from langextract_bedrock.schema import BedrockToolUseSchema

    # Create examples with various attribute types
    examples = [
        lx.data.ExampleData(
            text="Test document",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Test",
                    attributes={
                        "string_attr": "value",
                        "list_attr": ["a", "b", "c"],  # This should become string
                        "number_attr": "123",
                    },
                ),
            ],
        )
    ]

    schema_instance = BedrockToolUseSchema.from_examples(examples)
    schema_dict = schema_instance.to_provider_config()["schema"]

    # Get attribute properties
    items = schema_dict["properties"]["extractions"]["items"]
    attr_props = items["properties"].get("entity_attributes", {}).get(
        "properties", {}
    )

    # All attribute types should be "string" (not "array")
    for attr_name, attr_schema in attr_props.items():
        attr_type = attr_schema.get("type")
        assert (
            attr_type == "string"
        ), f"Attribute '{attr_name}' has type '{attr_type}', expected 'string'"

    print("✓ All attribute types are 'string'")
    print("  (Arrays are converted to comma-separated strings by sanitizer)")

    print("\n✅ TEST 4 PASSED")


def test_5_schema_dict_property():
    """Test 5: Verify schema_dict property works."""
    print_section("TEST 5: schema_dict property")

    from langextract_bedrock.schema import BedrockToolUseSchema

    examples = [
        lx.data.ExampleData(
            text="Test",
            extractions=[
                lx.data.Extraction(
                    extraction_class="test",
                    extraction_text="value",
                ),
            ],
        )
    ]

    schema_instance = BedrockToolUseSchema.from_examples(examples)

    # Verify schema_dict property exists and returns correct data
    schema_dict = schema_instance.schema_dict

    assert isinstance(
        schema_dict, dict
    ), f"schema_dict returns {type(schema_dict)}, expected dict"
    assert (
        "extractions" in schema_dict.get("properties", {})
    ), "schema_dict missing extractions property"

    print("✓ schema_dict property returns correct schema")
    print("\n✅ TEST 5 PASSED")


# =============================================================================
# INTEGRATION TESTS (AWS Required)
# =============================================================================


def test_6_extraction_with_fence_output():
    """Test 6: Verify extraction with fence_output=True (text mode)."""
    print_section("TEST 6: Extraction with fence_output=True")

    print("⚠️  This test requires valid AWS credentials")
    print("⏳ Running extraction in text mode (fence_output=True)...")

    prompt = textwrap.dedent("""\
        Extract character names from the text.
        Respond with ```json and end with ```.
    """)

    examples = [
        lx.data.ExampleData(
            text="ROMEO spoke to JULIET.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"family": "Montague"},
                )
            ],
        )
    ]

    result = lx.extract(
        text_or_documents="ROMEO and JULIET met at the ball.",
        prompt_description=prompt,
        examples=examples,
        model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        use_schema_constraints=True,
        fence_output=True,  # Text mode
        temperature=0.0,
        max_workers=1,
    )

    print(f"✓ Extraction completed: {len(result.extractions)} extractions")

    for i, ext in enumerate(result.extractions[:3]):
        print(f"  {i+1}. {ext.extraction_class}: '{ext.extraction_text}'")

    # Verify all extraction_text are strings
    for ext in result.extractions:
        assert isinstance(
            ext.extraction_text, str
        ), f"extraction_text is not string: {type(ext.extraction_text)}"

    print("✓ All extraction_text are strings")
    print("\n✅ TEST 6 PASSED")


def test_7_extraction_with_tool_use():
    """Test 7: Verify extraction with Tool Use API (fence_output=False)."""
    print_section("TEST 7: Extraction with Tool Use API")

    print("⚠️  This test requires valid AWS credentials")
    print("⏳ Running extraction in Tool Use mode (fence_output=False)...")

    prompt = "Extract character names from Shakespeare text."

    examples = [
        lx.data.ExampleData(
            text="ROMEO: But soft!",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"family": "Montague"},
                ),
            ],
        )
    ]

    document_text = textwrap.dedent("""\
        ROMEO: But soft, what light through yonder window breaks?
        JULIET: O Romeo, Romeo, wherefore art thou Romeo?
    """)

    result = lx.extract(
        text_or_documents=document_text,
        prompt_description=prompt,
        examples=examples,
        model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        use_schema_constraints=True,
        fence_output=False,  # Tool Use mode
        temperature=0.0,
        max_workers=1,
    )

    print(f"✓ Extraction completed: {len(result.extractions)} extractions")

    # Verify all extraction_text are strings
    for ext in result.extractions:
        assert isinstance(
            ext.extraction_text, str
        ), f"extraction_text is not string: {type(ext.extraction_text)}"

    print("✓ All extraction_text are strings")

    # Verify attributes have primitive values
    for ext in result.extractions:
        if ext.attributes:
            for key, value in ext.attributes.items():
                assert isinstance(
                    value, (str, int, float)
                ), f"Attribute '{key}' has non-primitive value: {type(value)}"

    print("✓ All attributes have primitive values")

    # Show results
    print("\n📊 Extractions obtained:")
    for i, ext in enumerate(result.extractions, 1):
        print(f"  {i}. {ext.extraction_class}: '{ext.extraction_text}'")

    print("\n✅ TEST 7 PASSED")


def test_8_long_document():
    """Test 8: Verify with long document and multiple chunks."""
    print_section("TEST 8: Long document processing")

    print("⚠️  This test requires valid AWS credentials")
    print("⏳ Processing long document...")

    prompt = textwrap.dedent("""\
        Extract judicial information from text.

        FIELDS:
        - numero_fallo: Case numbers
        - entidades: Organizations mentioned
        - personas: People mentioned with their roles
    """)

    examples = [
        lx.data.ExampleData(
            text="CASE N° 88/2025. Ministry submitted documents.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="numero_fallo",
                    extraction_text="CASE N° 88/2025",
                    attributes={"tipo": "Sentencia"},
                ),
                lx.data.Extraction(
                    extraction_class="entidades",
                    extraction_text="Ministry",
                    attributes={"sector": "Government"},
                ),
            ],
        )
    ]

    # Long document (repeat to increase size)
    document_text = """
    DOCUMENT N° 88/2025
    Viedma, June 17, 2025

    The Provincial Health Council of Pepito Fernandez Province submitted
    account statements for period 09/01/2023 to 09/30/2023.

    Officials:
    - Health Minister: Luis zzz.
    - Planning Secretary: Dana Nnonono
    - Management Secretary: Natali Sisisisi

    The Court of Accounts participated in the audit process.
    """ * 3  # Repeat 3 times

    result = lx.extract(
        text_or_documents=document_text,
        prompt_description=prompt,
        examples=examples,
        model_id="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        use_schema_constraints=True,
        fence_output=True,
        temperature=0.0,
        max_workers=3,
        max_char_buffer=2000,
    )

    print(f"✓ Document processed: {len(document_text)} characters")
    print(f"✓ Extractions: {len(result.extractions)}")

    # Count by type
    by_class: dict[str, int] = {}
    for ext in result.extractions:
        by_class[ext.extraction_class] = by_class.get(ext.extraction_class, 0) + 1

    print("\n📊 Extractions by class:")
    for cls, count in by_class.items():
        print(f"  - {cls}: {count}")

    # Verify no format errors
    for i, ext in enumerate(result.extractions):
        assert isinstance(
            ext.extraction_text, str
        ), f"Extraction #{i}: extraction_text is not string"
        if ext.attributes:
            for k, v in ext.attributes.items():
                assert not isinstance(
                    v, (list, dict)
                ), f"Extraction #{i}: attribute '{k}' is {type(v)}"

    print("✓ All formats are correct")
    print("\n✅ TEST 8 PASSED")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  TESTING LANGEXTRACT-BEDROCK SCHEMA AND PROVIDER")
    print("=" * 70)

    # Define tests: (name, function, requires_aws)
    tests = [
        # Unit tests (no AWS)
        ("Schema Class Import", test_1_schema_class_exists, False),
        ("get_schema_class() Method", test_2_get_schema_class, False),
        ("Dynamic Properties Schema", test_3_schema_generation_dynamic_properties, False),
        ("Attribute Types (Strings Only)", test_4_schema_attribute_types, False),
        ("schema_dict Property", test_5_schema_dict_property, False),
        # Integration tests (AWS required)
        ("Extraction (fence_output=True)", test_6_extraction_with_fence_output, True),
        ("Extraction (Tool Use API)", test_7_extraction_with_tool_use, True),
        ("Long Document Processing", test_8_long_document, True),
    ]

    results: list[tuple[str, bool | None]] = []

    for name, test_fn, requires_aws in tests:
        if requires_aws:
            response = input(f"\n¿Run '{name}' (requires AWS)? [y/N]: ").strip().lower()
            if response not in ["y", "yes", "s", "si"]:
                print(f"⏭️  Skipping {name}")
                results.append((name, None))
                continue

        success = test_fn()
        results.append((name, success))

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        if result is True:
            print(f"✅ {name}")
        elif result is False:
            print(f"❌ {name}")
        else:
            print(f"⏭️  {name} (skipped)")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n⚠️  Some tests failed. Review errors above.")
        sys.exit(1)
    elif passed == 0:
        print("\n⚠️  No tests ran.")
        sys.exit(0)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
