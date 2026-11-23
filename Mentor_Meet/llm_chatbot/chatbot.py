
import logging
import requests

from llm_client import get_response

logger = logging.getLogger(__name__)


class ChatBot:
    """Manage chatbot conversations and interactions with the LLM."""

    def __init__(self, max_history_messages: int = 12) -> None:
        """Initialize the chatbot and conversation history."""
        self.messages: list[dict[str, str]] = []
        self.max_history_messages = max_history_messages

        if self.max_history_messages < 2:
            raise ValueError("max_history_messages must be at least 2.")

    def chat(self, user_input: str) -> str:
        """Send a user message to the LLM and return its response."""
        user_input = user_input.strip()

        if not user_input:
            raise ValueError("Message cannot be empty.")

        self.messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            response = get_response(self.messages)

            if not isinstance(response, str) or not response.strip():
                raise RuntimeError("The model returned an empty response.")

            self.messages.append({
                "role": "assistant",
                "content": response,
            })

            self._trim_history()
            return response.strip()

        except (requests.RequestException, RuntimeError, ValueError):
            # Remove the current user message if the API request fails.
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            logger.exception("Failed to generate a chatbot response.")
            raise

    def reset_conversation(self) -> None:
        """Clear the conversation history."""
        self.messages.clear()
        logger.info("Conversation history reset.")

    def _trim_history(self) -> None:
        """Keep recent messages without splitting user-assistant turns."""
        max_messages = self.max_history_messages

        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

            # Ensure history begins with a user message.
            while self.messages and self.messages[0]["role"] != "user":
                self.messages.pop(0)
