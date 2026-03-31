#!/usr/bin/env python3
"""
Multi-turn chatbot — supports OpenRouter (multi-provider) or direct xAI API.

Via OpenRouter (Anthropic, OpenAI, xAI through one key):
    export OPENROUTER_API_KEY=sk-or-...
    python chatbot.py --model claude           # Anthropic Claude Opus 4.6
    python chatbot.py --model sonnet           # Anthropic Claude Sonnet 4.6
    python chatbot.py --model gpt4             # OpenAI GPT-4o
    python chatbot.py --model grok             # xAI Grok (via OpenRouter)
    python chatbot.py --model anthropic/claude-opus-4.6  # full OpenRouter model ID

Via xAI directly (your xAI API key):
    export XAI_API_KEY=xai-...
    python chatbot.py --provider xai                     # default: grok-2
    python chatbot.py --provider xai --model grok-2
    python chatbot.py --provider xai --model grok-2-mini
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI, APIError

# ── Providers ─────────────────────────────────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var":  "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-opus-4.6",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "env_var":  "XAI_API_KEY",
        "default_model": "grok-2",
    },
}

# ── Model aliases (OpenRouter) ────────────────────────────────────────────────

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
    parser = argparse.ArgumentParser(description="Multi-provider chatbot")
    parser.add_argument(
        "--provider", "-p",
        default="openrouter",
        choices=list(PROVIDERS),
        help="API provider to use (default: openrouter)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help=(
            "Model alias or full model ID. "
            f"OpenRouter aliases: {', '.join(MODEL_ALIASES)}. "
            "xAI models: grok-2, grok-2-mini. "
            "Defaults to provider's default model."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = PROVIDERS[args.provider]

    # Resolve model: CLI arg > alias expansion > provider default
    raw_model = args.model or provider["default_model"]
    model = MODEL_ALIASES.get(raw_model, raw_model)

    api_key = os.environ.get(provider["env_var"])
    if not api_key:
        print(f"Error: {provider['env_var']} environment variable is not set.")
        if args.provider == "xai":
            print("Get a key at https://console.x.ai/")
        else:
            print("Get a key at https://openrouter.ai/settings/keys")
        sys.exit(1)

    client = OpenAI(
        base_url=provider["base_url"],
        api_key=api_key,
    )

    system = os.environ.get("CHATBOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    messages: list = []
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"Chatbot — model: {model} (via {args.provider})")
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
