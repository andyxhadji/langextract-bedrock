"""
Test script for verifying schema constraints support in langextract-bedrock.

Run: python test_schema_constraints.py

Requirements:
- AWS credentials configured
- langextract installed
- langextract-bedrock installed in editable mode (pip install -e .)
"""

import langextract as lx
import textwrap
import json
import sys

from dotenv import load_dotenv

load_dotenv()


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_1_schema_class_exists():
    """Test 1: Verify BedrockToolUseSchema exists and is correct type."""
    print_section("TEST 1: Verify schema class exists")
    
    try:
        from langextract_bedrock.provider import BedrockToolUseSchema
        
        # Verify it's a subclass of BaseSchema
        if not issubclass(BedrockToolUseSchema, lx.schema.BaseSchema):
            print(f"❌ FAIL: BedrockToolUseSchema is not a BaseSchema subclass")
            return False
        
        print("✓ BedrockToolUseSchema exists and extends BaseSchema")
        print("\n✅ TEST 1 PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ TEST 1 FAILED: Cannot import BedrockToolUseSchema: {e}")
        return False
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_get_schema_class():
    """Test 2: Verify get_schema_class() returns BedrockToolUseSchema."""
    print_section("TEST 2: Verify get_schema_class() method")
    
    try:
        from langextract_bedrock.provider import BedrockLanguageModel, BedrockToolUseSchema
        
        # Check class method
        schema_class = BedrockLanguageModel.get_schema_class()
        
        if schema_class is None:
            print("❌ FAIL: get_schema_class() returns None")
            return False
        
        if schema_class != BedrockToolUseSchema:
            print(f"❌ FAIL: get_schema_class() returns {schema_class}, expected BedrockToolUseSchema")
            return False
        
        print("✓ get_schema_class() returns BedrockToolUseSchema")
        print("\n✅ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_schema_generation():
    """Test 3: Verify schema generation from examples."""
    print_section("TEST 3: Schema generation from examples")
    
    try:
        from langextract_bedrock.provider import BedrockToolUseSchema
        
        # Create test examples using Romeo and Juliet
        examples = [
            lx.data.ExampleData(
                text="ROMEO: But soft, what light through yonder window breaks?",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="character",
                        extraction_text="ROMEO",
                        attributes={"family": "Montague"}
                    ),
                    lx.data.Extraction(
                        extraction_class="dialogue",
                        extraction_text="But soft, what light through yonder window breaks?",
                        attributes={"tone": "wonder"}
                    ),
                    lx.data.Extraction(
                        extraction_class="emotion",
                        extraction_text="soft",
                        attributes={"intensity": "gentle"}
                    )
                ]
            )
        ]
        
        # Generate schema
        schema_instance = BedrockToolUseSchema.from_examples(examples)
        
        # Verify it's an instance of BedrockToolUseSchema
        if not isinstance(schema_instance, BedrockToolUseSchema):
            print(f"❌ FAIL: from_examples() returned {type(schema_instance)}")
            return False
        
        print("✓ from_examples() returns BedrockToolUseSchema instance")
        
        # Check to_provider_config() returns dict with schema
        config = schema_instance.to_provider_config()
        
        if not isinstance(config, dict):
            print(f"❌ FAIL: to_provider_config() returns {type(config)}, expected dict")
            return False
        
        if "schema" not in config:
            print(f"❌ FAIL: to_provider_config() missing 'schema' key. Keys: {config.keys()}")
            return False
        
        print("✓ to_provider_config() returns dict with 'schema' key")
        
        # Verify schema structure
        schema_dict = config["schema"]
        
        if "properties" not in schema_dict or "extractions" not in schema_dict["properties"]:
            print("❌ FAIL: Schema missing expected structure")
            print(json.dumps(schema_dict, indent=2))
            return False
        
        print("✓ Schema has correct structure")
        
        # Verify extraction classes in enum
        items = schema_dict["properties"]["extractions"]["items"]
        enum_values = items["properties"]["extraction_class"]["enum"]
        
        expected = {"character", "dialogue", "emotion"}
        actual = set(enum_values)
        
        if expected != actual:
            print(f"❌ FAIL: Extraction classes mismatch")
            print(f"   Expected: {expected}")
            print(f"   Actual: {actual}")
            return False
        
        print(f"✓ Extraction classes correct: {actual}")
        
        # Check requires_raw_output
        if not schema_instance.requires_raw_output:
            print("❌ FAIL: requires_raw_output should be True for Tool Use")
            return False
        
        print("✓ requires_raw_output is True")
        
        # Show generated schema
        print("\n📋 Generated schema:")
        print(json.dumps(schema_dict, indent=2, ensure_ascii=False)[:500] + "...")
        
        print("\n✅ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_backward_compatibility():
    """Test 4: Verify existing code (without schema) still works."""
    print_section("TEST 4: Backward compatibility")
    
    print("⚠️  This test requires valid AWS credentials")
    print("⏳ Running extraction without schema constraints...")
    
    try:
        prompt = "Extract character names from the text."
        examples = [
            lx.data.ExampleData(
                text="ROMEO spoke to JULIET by the window.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="character",
                        extraction_text="ROMEO",
                        attributes={"family": "Montague"}
                    )
                ]
            )
        ]
        
        # Legacy mode (as before the changes)
        result = lx.extract(
            text_or_documents="ROMEO and JULIET met at the ball. The NURSE watched from afar.",
            prompt_description=prompt,
            examples=examples,
            model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            use_schema_constraints=False,
            temperature=0.0,
            max_workers=1
        )
        
        print(f"✓ Extraction completed: {len(result.extractions)} extractions")
        
        for i, ext in enumerate(result.extractions[:3]):
            print(f"  {i+1}. {ext.extraction_class}: '{ext.extraction_text}'")
        
        print("\n✅ TEST 4 PASSED - Backward compatibility OK")
        return True
        
    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_schema_constraints_enabled():
    """Test 5: Verify use_schema_constraints=True works (NEW FEATURE)."""
    print_section("TEST 5: Schema constraints enabled (NEW FEATURE)")
    
    print("⚠️  This test requires valid AWS credentials")
    print("⏳ Running extraction WITH schema constraints...")
    
    try:
        prompt = textwrap.dedent("""\
            Extract information from Shakespeare's Romeo and Juliet.
            
            FIELDS:
            - character: Character names as they appear
            - location: Places mentioned in the scene
            - emotion: Emotional expressions or states
            - relationship: Relationships between characters
        """)
        
        examples = [
            lx.data.ExampleData(
                text="ROMEO: But soft, what light through yonder window breaks? It is the east, and Juliet is the sun.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="character",
                        extraction_text="ROMEO",
                        attributes={"family": "Montague"}
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="window",
                        attributes={"type": "architectural"}
                    ),
                    lx.data.Extraction(
                        extraction_class="emotion",
                        extraction_text="soft",
                        attributes={"intensity": "gentle"}
                    ),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="Juliet is the sun",
                        attributes={"type": "metaphor"}
                    )
                ]
            )
        ]
        
        document_text = textwrap.dedent("""\
            Act 2, Scene 2: Capulet's Orchard
            
            [Enter ROMEO]
            
            ROMEO: He jests at scars that never felt a wound.
            But soft, what light through yonder window breaks?
            It is the east, and Juliet is the sun.
            Arise, fair sun, and kill the envious moon.
            
            [JULIET appears above at a window]
            
            JULIET: O Romeo, Romeo, wherefore art thou Romeo?
            Deny thy father and refuse thy name.
            Or if thou wilt not, be but sworn my love,
            And I'll no longer be a Capulet.
        """)
        
        # THE NEW FEATURE - schema constraints enabled!
        result = lx.extract(
            text_or_documents=document_text,
            prompt_description=prompt,
            examples=examples,
            model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            use_schema_constraints=True,  # ✅ NEW: Works now!
            temperature=0.0,
            max_workers=1
        )
        
        print(f"✓ Extraction completed: {len(result.extractions)} extractions")
        
        # Verify all extraction_text are strings
        for ext in result.extractions:
            if not isinstance(ext.extraction_text, str):
                print(f"❌ FAIL: extraction_text is not string: {type(ext.extraction_text)}")
                return False
        
        print("✓ All extraction_text are strings")
        
        # Verify attributes have primitive values
        for ext in result.extractions:
            if ext.attributes:
                for key, value in ext.attributes.items():
                    if not isinstance(value, (str, int, float)):
                        print(f"❌ FAIL: Attribute '{key}' has non-primitive value: {type(value)}")
                        return False
        
        print("✓ All attributes have primitive values")
        
        # Show results
        print("\n📊 Extractions obtained:")
        for i, ext in enumerate(result.extractions, 1):
            print(f"\n{i}. Class: {ext.extraction_class}")
            print(f"   Text: '{ext.extraction_text}'")
            if ext.attributes:
                print(f"   Attributes: {ext.attributes}")
        
        print("\n✅ TEST 5 PASSED - Schema constraints works!")
        return True
        
    except Exception as e:
        print(f"❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_long_document():
    """Test 6: Verify with long document (the original failing case)."""
    print_section("TEST 6: Long document (real-world case)")
    
    print("⚠️  This test requires valid AWS credentials")
    print("⏳ Processing long document...")
    
    try:
        prompt = textwrap.dedent("""\
            Extract judicial information from text.
            
            RULES:
            - extraction_text: always a simple string
            - For multiple values: create separate objects
            - attributes: only string/number values
        """)
        
        examples = [
            lx.data.ExampleData(
                text="CASE N° 88/2025. Ministry and Council submitted documents.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="numero_fallo",
                        extraction_text="CASE N° 88/2025",
                        attributes={"tipo": "Sentencia"}
                    ),
                    lx.data.Extraction(
                        extraction_class="entidades",
                        extraction_text="Ministry",
                        attributes={"tipo": "Organismo"}
                    ),
                    lx.data.Extraction(
                        extraction_class="entidades",
                        extraction_text="Council",
                        attributes={"tipo": "Organismo"}
                    )
                ]
            )
        ]
        
        # Long document (repeat to increase size)
        document_text = """
        FALLO TCRN N° 88/2025
        Viedma, June 17, 2025
        
        In compliance with article 1 of the Annex to Resolution "T" No 02/2009,
        the Provincial Health Council of Río Negro Province submitted to this
        external control body the account statements for period 09/01/2023 to 09/30/2023.
        
        ARTICLE 1: Approve the accounts submitted by the Provincial Health Council
        for the mentioned period.
        
        Officials of the Provincial Health Council, according to administrative acts:
        
        - Health Minister and President: Luis F. ZGAIB
        - Planning Secretary: Daiana Noelia BECKER
        - Institutional Relations Secretary: Miguel Angel LEDESMA
        - Management Secretary: Natali CAMBRUZZI
        - Strategies Secretary: Martín Miguel CILIBERTO
        
        The Health Ministry of Río Negro Province and the Court of Accounts
        participated in the audit process.
        """ * 3  # Repeat 3 times to make longer
        
        result = lx.extract(
            text_or_documents=document_text,
            prompt_description=prompt,
            examples=examples,
            model_id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            use_schema_constraints=True,
            temperature=0.0,
            max_workers=5,
            max_char_buffer=2000
        )
        
        print(f"✓ Document processed: {len(document_text)} characters")
        print(f"✓ Extractions: {len(result.extractions)}")
        
        # Count by type
        by_class = {}
        for ext in result.extractions:
            by_class[ext.extraction_class] = by_class.get(ext.extraction_class, 0) + 1
        
        print("\n📊 Extractions by class:")
        for cls, count in by_class.items():
            print(f"  - {cls}: {count}")
        
        # Verify no format errors
        errors = []
        for i, ext in enumerate(result.extractions):
            if not isinstance(ext.extraction_text, str):
                errors.append(f"Extraction #{i}: extraction_text is not string")
            if ext.attributes:
                for k, v in ext.attributes.items():
                    if isinstance(v, (list, dict)):
                        errors.append(f"Extraction #{i}: attribute '{k}' is {type(v)}")
        
        if errors:
            print("\n❌ Format errors found:")
            for err in errors:
                print(f"  - {err}")
            return False
        
        print("✓ All formats are correct")
        
        print("\n✅ TEST 6 PASSED - Long documents work!")
        return True
        
    except Exception as e:
        print(f"❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  TESTING SCHEMA CONSTRAINTS FOR LANGEXTRACT-BEDROCK")
    print("=" * 70)
    
    tests = [
        ("Schema Class Exists", test_1_schema_class_exists, False),
        ("get_schema_class() Method", test_2_get_schema_class, False),
        ("Schema Generation", test_3_schema_generation, False),
        ("Backward Compatibility", test_4_backward_compatibility, True),
        ("Schema Constraints", test_5_schema_constraints_enabled, True),
        ("Long Document", test_6_long_document, True),
    ]
    
    results = []
    
    for name, test_fn, requires_aws in tests:
        if requires_aws:
            response = input(f"\n¿Run '{name}' (requires AWS)? [y/N]: ").strip().lower()
            if response not in ['y', 'yes', 's', 'si']:
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
        print("\n⚠️  No AWS tests ran. For full testing, run with credentials.")
        sys.exit(0)
    else:
        print("\n🎉 All tests passed! Implementation works correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()