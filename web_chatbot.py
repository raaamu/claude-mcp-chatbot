#!/usr/bin/env python3
"""
Web chatbot with streaming, Gmail MCP tools, and Gradio UI.
Supports OpenRouter (Anthropic/OpenAI/xAI) and direct xAI API.

Usage:
    # OpenRouter (default — Claude Opus 4.6)
    export OPENROUTER_API_KEY=sk-or-...
    python web_chatbot.py

    # xAI directly
    export XAI_API_KEY=xai-...
    python web_chatbot.py --provider xai

    # Specific model
    python web_chatbot.py --provider xai --model grok-2
    python web_chatbot.py --model sonnet
"""

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Generator

import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI, APIError

SERVER_SCRIPT = str(Path(__file__).parent / "gmail_mcp_server.py")

SYSTEM_PROMPT = os.environ.get(
    "CHATBOT_SYSTEM_PROMPT",
    "You are a helpful assistant with access to Gmail. "
    "You can search, read, and summarize emails when asked. "
    "When reading emails, be concise — summarize unless the user asks for the full text. "
    "If a request is ambiguous, ask a clarifying question.",
)

# ── Providers & model aliases ─────────────────────────────────────────────────

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

MODEL_ALIASES: dict[str, str] = {
    "claude":  "anthropic/claude-opus-4.6",
    "sonnet":  "anthropic/claude-sonnet-4.6",
    "gpt4":    "openai/gpt-4o",
    "grok":    "x-ai/grok-4.20-beta",
}


# ── MCP client (persistent background thread) ────────────────────────────────

class MCPClient:
    """Manages a persistent MCP server session in a dedicated event-loop thread."""

    def __init__(self, server_script: str):
        self._server_script = server_script
        self.tools: list[dict] = []          # stored in Anthropic input_schema format
        self.status = "not started"
        self._session: ClientSession | None = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-thread")

    def start(self) -> bool:
        self._thread.start()
        self._ready.wait(timeout=45)
        return bool(self._session)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())

    async def _connect(self):
        params = StdioServerParameters(command=sys.executable, args=[self._server_script])
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    self.tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "input_schema": (
                                t.inputSchema
                                if isinstance(t.inputSchema, dict)
                                else t.inputSchema.model_dump()
                            ),
                        }
                        for t in result.tools
                    ]
                    self._session = session
                    self.status = f"{len(self.tools)} tools loaded"
                    self._ready.set()
                    await asyncio.sleep(float("inf"))
        except Exception as e:
            self.status = f"failed: {e}"
            self._ready.set()

    def openai_tools(self) -> list[dict]:
        """Return tools in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in self.tools
        ]

    def call_tool(self, name: str, tool_input: dict) -> str:
        if not self._session:
            return "[MCP not connected — Gmail tools unavailable]"
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(name, tool_input),
                self._loop,
            )
            result = future.result(timeout=60)
            return "\n".join(
                block.text if hasattr(block, "text") else str(block)
                for block in result.content
            )
        except Exception as e:
            return f"[Tool error: {e}]"


# ── Chat logic ────────────────────────────────────────────────────────────────

def respond(
    message: str,
    chatbot: list[dict],
    msgs: list[dict],
) -> Generator[tuple[list[dict], list[dict]], None, None]:
    """Streaming chat generator. Yields (chatbot, msgs) on every token."""

    if not message.strip():
        yield chatbot, msgs
        return

    chatbot = list(chatbot) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    msgs = list(msgs) + [{"role": "user", "content": message}]
    yield chatbot, msgs

    openai_tools = mcp.openai_tools()
    display_text = ""

    try:
        while True:
            api_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

            stream = llm_client.chat.completions.create(
                model=model,
                messages=api_msgs,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                stream=True,
                stream_options={"include_usage": True},
            )

            chunk_text = ""
            tool_calls_buf: dict[int, dict] = {}  # index → accumulated tool call
            finish_reason = None

            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta

                if delta.content:
                    chunk_text += delta.content
                    chatbot[-1]["content"] = display_text + chunk_text
                    yield chatbot, msgs

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buf:
                            tool_calls_buf[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_buf[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_buf[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_buf[idx]["arguments"] += tc.function.arguments

            display_text += chunk_text

            if finish_reason != "tool_calls":
                chatbot[-1]["content"] = display_text
                msgs = msgs + [{"role": "assistant", "content": display_text}]
                yield chatbot, msgs
                break

            # Build the assistant message with tool_calls
            tool_calls_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls_buf.values()
            ]
            msgs = msgs + [{
                "role": "assistant",
                "content": chunk_text or None,
                "tool_calls": tool_calls_list,
            }]

            # Execute each tool and collect results
            for tc in tool_calls_list:
                name = tc["function"]["name"]
                try:
                    tool_input = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_input = {}

                notice = f"\n\n**[Tool: {name}]**\n"
                display_text += notice
                chatbot[-1]["content"] = display_text
                yield chatbot, msgs

                result = mcp.call_tool(name, tool_input)
                msgs = msgs + [{
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }]

    except APIError as e:
        chatbot[-1]["content"] = display_text + f"\n\n**[API error: {e}]**"
        yield chatbot, msgs


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    tool_count = len(mcp.tools)
    mcp_badge = (
        f"Gmail connected — {tool_count} tools available"
        if tool_count
        else f"Gmail unavailable ({mcp.status})"
    )

    with gr.Blocks(title="Chatbot + Gmail") as demo:
        gr.Markdown("## Chatbot + Gmail Assistant")
        gr.Markdown(
            f"**Model:** `{model}` &nbsp;|&nbsp; "
            f"**Provider:** {provider_name} &nbsp;|&nbsp; "
            f"**MCP:** {mcp_badge}"
        )

        msgs_state = gr.State([])

        chatbot = gr.Chatbot(
            height=560,
            buttons=["copy", "copy_all"],
            label="Conversation",
            render_markdown=True,
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask about your emails, or just chat…",
                scale=8,
                show_label=False,
                autofocus=True,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")
            clear_btn = gr.Button("Clear", scale=1, variant="secondary")

        with gr.Accordion("System prompt", open=False):
            gr.Markdown(f"```\n{SYSTEM_PROMPT}\n```")

        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, msgs_state],
            outputs=[chatbot, msgs_state],
        ).then(lambda: gr.update(value=""), outputs=msg_box)

        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, msgs_state],
            outputs=[chatbot, msgs_state],
        ).then(lambda: gr.update(value=""), outputs=msg_box)

        clear_btn.click(lambda: ([], []), outputs=[chatbot, msgs_state])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web chatbot with Gmail MCP")
    parser.add_argument(
        "--provider", "-p",
        default="openrouter",
        choices=list(PROVIDERS),
        help="API provider (default: openrouter)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model alias or full model ID (default: provider's default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    provider_name = args.provider
    provider_cfg = PROVIDERS[provider_name]

    raw_model = args.model or provider_cfg["default_model"]
    model = MODEL_ALIASES.get(raw_model, raw_model)

    api_key = os.environ.get(provider_cfg["env_var"])
    if not api_key:
        print(f"Error: {provider_cfg['env_var']} is not set.")
        if provider_name == "xai":
            print("Get a key at https://console.x.ai/")
        else:
            print("Get a key at https://openrouter.ai/settings/keys")
        sys.exit(1)

    llm_client = OpenAI(base_url=provider_cfg["base_url"], api_key=api_key)

    print("Connecting to Gmail MCP server…")
    mcp = MCPClient(SERVER_SCRIPT)
    ok = mcp.start()
    if ok:
        print(f"Gmail MCP ready — {mcp.status}")
    else:
        print(f"Gmail MCP unavailable ({mcp.status}) — continuing without email tools")

    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
