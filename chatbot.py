#!/usr/bin/env python3
"""
Multi-turn chatbot via OpenRouter — supports Anthropic, OpenAI, and xAI.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python chatbot.py                          # default model (Claude Opus 4.6)
    python chatbot.py --model claude           # Anthropic Claude Opus 4.6
    python chatbot.py --model sonnet           # Anthropic Claude Sonnet 4.6
    python chatbot.py --model gpt4            # OpenAI GPT-4o
    python chatbot.py --model grok            # xAI Grok
    python chatbot.py --model anthropic/claude-opus-4.6  # full OpenRouter model ID
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI, APIError

# ── Model aliases ─────────────────────────────────────────────────────────────

MODEL_ALIASES: dict[str, str] = {
    "claude":  "anthropic/claude-opus-4.6",
    "sonnet":  "anthropic/claude-sonnet-4.6",
    "gpt4":    "openai/gpt-4o",
    "grok":    "x-ai/grok-4.20-beta",
}
DEFAULT_MODEL = "anthropic/claude-opus-4.6"

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
  /model   — Show current model
  /system  — Show current system prompt
  /help    — Show this help
  /quit    — Exit
"""


# ── Streaming ─────────────────────────────────────────────────────────────────

def stream_response(
    client: OpenAI, model: str, messages: list, system: str
) -> tuple[str, int, int]:
    """Stream a response and return (full_text, input_tokens, output_tokens)."""
    full_text = ""
    input_tokens = 0
    output_tokens = 0

    all_messages = [{"role": "system", "content": system}] + messages

    stream = client.chat.completions.create(
        model=model,
        messages=all_messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            full_text += delta.content
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens

    print()  # newline after streamed response
    return full_text, input_tokens, output_tokens


# ── Save / load ───────────────────────────────────────────────────────────────

def save_history(messages: list, system: str, model: str) -> None:
    data = {
        "saved_at": datetime.now().isoformat(),
        "model": model,
        "system": system,
        "messages": messages,
    }
    HISTORY_FILE.write_text(json.dumps(data, indent=2))
    print(f"[Saved to {HISTORY_FILE}]")


def load_history() -> tuple[list, str | None, str | None]:
    if not HISTORY_FILE.exists():
        print(f"[No history file found at {HISTORY_FILE}]")
        return [], None, None
    data = json.loads(HISTORY_FILE.read_text())
    print(f"[Loaded {len(data['messages'])} messages from {data['saved_at']}]")
    return data["messages"], data.get("system"), data.get("model")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-provider chatbot via OpenRouter")
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=(
            f"Model alias ({', '.join(MODEL_ALIASES)}) "
            "or full OpenRouter model ID (default: %(default)s)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MODEL_ALIASES.get(args.model, args.model)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        print("Get a key at https://openrouter.ai/settings/keys")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    system = os.environ.get("CHATBOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    messages: list = []
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"Chatbot — model: {model} (via OpenRouter)")
    print(f"System:  {system}")
    print(COMMANDS)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

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
                save_history(messages, system, model)
            elif cmd == "/load":
                loaded_messages, loaded_system, loaded_model = load_history()
                messages = loaded_messages
                if loaded_system:
                    system = loaded_system
                    print(f"[System prompt restored: {system}]")
                if loaded_model:
                    model = loaded_model
                    print(f"[Model restored: {model}]")
            elif cmd == "/model":
                print(f"[Model: {model}]")
            elif cmd == "/system":
                print(f"[System: {system}]")
            elif cmd == "/help":
                print(COMMANDS)
            else:
                print(f"[Unknown command: {user_input}]")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        try:
            response_text, input_tokens, output_tokens = stream_response(
                client, model, messages, system
            )
        except APIError as e:
            print(f"\n[API error: {e}]")
            messages.pop()
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
