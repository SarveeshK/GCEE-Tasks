
import logging
import requests

from config import (
    API_TOKEN,
    BASE_URL,
    MODEL_NAME,
    SYSTEM_PROMPT,
    MAX_RESPONSE_TOKENS,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)


def get_response(messages: list[dict[str, str]]) -> str:
    """
    Send conversation messages to OpenRouter and return
    the model's text response.
    """
    if not API_TOKEN:
        raise ValueError(
            "API_KEY is missing. Check your .env file."
        )

    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *messages,
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": conversation,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": TEMPERATURE,
        "reasoning": {
            "enabled": False
        },
    }

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        logger.exception("Could not connect to OpenRouter.")
        raise RuntimeError(
            f"Could not connect to OpenRouter: {exc}"
        ) from exc

    if not response.ok:
        logger.error(
            "OpenRouter returned HTTP %s.",
            response.status_code,
        )
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: "
            f"{response.text}"
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "OpenRouter returned an invalid JSON response."
        ) from exc

    choices = result.get("choices", [])

    if not choices:
        raise RuntimeError(
            f"No response choices returned: {result}"
        )

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content")

    # Standard text response.
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Some APIs may return content as a list of text blocks.
    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
        ]

        combined_text = "\n".join(text_parts).strip()

        if combined_text:
            return combined_text

    finish_reason = choice.get("finish_reason")
    logger.warning(
        "Model returned no text. Finish reason: %s",
        finish_reason,
    )

    raise RuntimeError(
        "The model returned no text. "
        f"Finish reason: {finish_reason}. "
        "Try again or check the model's availability."
    )
