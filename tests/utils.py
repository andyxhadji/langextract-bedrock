import os


def aws_available() -> bool:
    return ("AWS_BEARER_TOKEN_BEDROCK" in os.environ) or (
        "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ
    )
