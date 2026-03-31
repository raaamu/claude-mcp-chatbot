#!/usr/bin/env python3
"""Multi-turn Claude chatbot with streaming, token tracking, and save/load."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-6"
HISTORY_FILE = Path("chat_history.json")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. "
    "Be concise but thorough. If you're unsure about something, say so."
)

COMMANDS = """
Commands:
  /clear   — Clear conversation history
  /save    — Save history to chat_history.json
  /load    — Load history from chat_history.json
  /system  — Show current system prompt
  /help    — Show this help
  /quit    — Exit
"""


def stream_response(client: anthropic.Anthropic, messages: list, system: str) -> tuple[str, int, int]:
    """Stream a response and return (full_text, input_tokens, output_tokens)."""
    full_text = ""
    input_tokens = 0
    output_tokens = 0

    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system=system,
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
                full_text += event.delta.text

        final = stream.get_final_message()
        input_tokens = final.usage.input_tokens
        output_tokens = final.usage.output_tokens

    print()  # newline after streamed response
    return full_text, input_tokens, output_tokens


def save_history(messages: list, system: str) -> None:
    data = {
        "saved_at": datetime.now().isoformat(),
        "system": system,
        "messages": messages,
    }
    HISTORY_FILE.write_text(json.dumps(data, indent=2))
    print(f"[Saved to {HISTORY_FILE}]")


def load_history() -> tuple[list, str | None]:
    if not HISTORY_FILE.exists():
        print(f"[No history file found at {HISTORY_FILE}]")
        return [], None
    data = json.loads(HISTORY_FILE.read_text())
    print(f"[Loaded {len(data['messages'])} messages from {data['saved_at']}]")
    return data["messages"], data.get("system")


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Allow overriding the system prompt via env var
    system = os.environ.get("CLAUDE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    messages: list = []

    total_input_tokens = 0
    total_output_tokens = 0

    print(f"Claude Chatbot ({MODEL})")
    print(f"System: {system}")
    print(COMMANDS)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit":
                print("Goodbye!")
                break
            elif cmd == "/clear":
                messages = []
                total_input_tokens = 0
                total_output_tokens = 0
                print("[Conversation cleared]")
            elif cmd == "/save":
                save_history(messages, system)
            elif cmd == "/load":
                loaded_messages, loaded_system = load_history()
                messages = loaded_messages
                if loaded_system:
                    system = loaded_system
                    print(f"[System prompt restored: {system}]")
            elif cmd == "/system":
                print(f"[System: {system}]")
            elif cmd == "/help":
                print(COMMANDS)
            else:
                print(f"[Unknown command: {user_input}]")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Claude: ", end="", flush=True)
        try:
            response_text, input_tokens, output_tokens = stream_response(
                client, messages, system
            )
        except anthropic.APIError as e:
            print(f"\n[API error: {e}]")
            messages.pop()  # remove the user message we just added
            continue

        messages.append({"role": "assistant", "content": response_text})

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        print(
            f"[tokens: {input_tokens} in / {output_tokens} out | "
            f"session total: {total_input_tokens} in / {total_output_tokens} out]"
        )


if __name__ == "__main__":
    main()
