
import logging

from chatbot import ChatBot


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


def main() -> None:
    """Run the interactive terminal chatbot."""
    configure_logging()

    try:
        chatbot = ChatBot()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return

    print("\nAI Assistant")
    print("-" * 32)
    print("Ask a question to get started.")
    print("Commands: /reset to clear history, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command in {"/exit", "exit", "quit"}:
                print("Assistant: Goodbye!")
                break

            if command == "/reset":
                chatbot.reset_conversation()
                print("Assistant: Conversation history cleared.\n")
                continue

            response = chatbot.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Session ended.")
            break

        except RuntimeError as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
