import os


def aws_available() -> bool:
    """Check if AWS credentials are available for integration tests."""
    has_bearer_token = "AWS_BEARER_TOKEN_BEDROCK" in os.environ
    has_aws_creds = (
        "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ
    )
    has_profile = "AWS_PROFILE" in os.environ
    has_region = "AWS_DEFAULT_REGION" in os.environ

    return has_bearer_token or has_aws_creds or has_profile or has_region