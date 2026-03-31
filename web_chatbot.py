#!/usr/bin/env python3
"""
Claude web chatbot with streaming, Gmail MCP tools, and Gradio UI.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python web_chatbot.py        # opens http://localhost:7860
"""

import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Generator

import anthropic
import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "claude-opus-4-6"
SYSTEM_PROMPT = os.environ.get(
    "CLAUDE_SYSTEM_PROMPT",
    "You are a helpful assistant with access to Gmail. "
    "You can search, read, and summarize emails when asked. "
    "When reading emails, be concise — summarize unless the user asks for the full text. "
    "If a request is ambiguous, ask a clarifying question.",
)
SERVER_SCRIPT = str(Path(__file__).parent / "gmail_mcp_server.py")

anthropic_client = anthropic.Anthropic()


# ── MCP client (persistent background thread) ────────────────────────────────

class MCPClient:
    """Manages a persistent MCP server session in a dedicated event-loop thread."""

    def __init__(self, server_script: str):
        self._server_script = server_script
        self.tools: list[dict] = []
        self.status = "not started"
        self._session: ClientSession | None = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-thread")

    def start(self) -> bool:
        """Start the MCP server. Returns True if connected successfully."""
        self._thread.start()
        self._ready.wait(timeout=45)
        return bool(self._session)

    # ── background thread ────────────────────────────────────────────────────

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())

    async def _connect(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=[self._server_script],
        )
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
                    await asyncio.sleep(float("inf"))   # keep alive
        except Exception as e:
            self.status = f"failed: {e}"
            self._ready.set()

    # ── called from Gradio thread ─────────────────────────────────────────────

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


# ── Content block helpers ─────────────────────────────────────────────────────

def _blocks_to_params(blocks) -> list[dict]:
    """Convert SDK ContentBlock objects → dict params safe to send in next turn."""
    result = []
    for b in blocks:
        if b.type == "text":
            result.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            result.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        elif b.type == "thinking":
            result.append({"type": "thinking", "thinking": b.thinking, "signature": b.signature})
        # redacted_thinking blocks are intentionally dropped
    return result


def _text_from_blocks(blocks) -> str:
    return "".join(b.text for b in blocks if b.type == "text")


# ── Chat logic ────────────────────────────────────────────────────────────────

def respond(
    message: str,
    chatbot: list[dict],
    claude_msgs: list[dict],
) -> Generator[tuple[list[dict], list[dict]], None, None]:
    """Streaming chat generator. Yields (chatbot, claude_msgs) on every token."""

    if not message.strip():
        yield chatbot, claude_msgs
        return

    # Append user turn to both display history and Claude history
    chatbot = list(chatbot) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    claude_msgs = list(claude_msgs) + [{"role": "user", "content": message}]

    yield chatbot, claude_msgs

    tools = mcp.tools
    display_text = ""  # accumulates everything shown in the current assistant bubble

    try:
        while True:
            stream_kwargs: dict = dict(
                model=MODEL,
                max_tokens=64000,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=claude_msgs,
            )
            if tools:
                stream_kwargs["tools"] = tools
                stream_kwargs["tool_choice"] = {"type": "auto"}

            with anthropic_client.messages.stream(**stream_kwargs) as stream:
                chunk = ""
                for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        chunk += event.delta.text
                        chatbot[-1]["content"] = display_text + chunk
                        yield chatbot, claude_msgs

                final = stream.get_final_message()
                display_text += chunk

            if final.stop_reason != "tool_use":
                # Natural end — commit assistant message as plain text
                chatbot[-1]["content"] = display_text
                claude_msgs = claude_msgs + [{"role": "assistant", "content": display_text}]
                yield chatbot, claude_msgs
                break

            # Tool use — preserve full content (thinking + tool_use blocks) for context
            claude_msgs = claude_msgs + [
                {"role": "assistant", "content": _blocks_to_params(final.content)}
            ]

            tool_results = []
            for block in final.content:
                if block.type != "tool_use":
                    continue

                notice = f"\n\n**[Tool: {block.name}]**\n"
                display_text += notice
                chatbot[-1]["content"] = display_text
                yield chatbot, claude_msgs

                tool_result = mcp.call_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result,
                })

            claude_msgs = claude_msgs + [{"role": "user", "content": tool_results}]

    except anthropic.APIError as e:
        chatbot[-1]["content"] = display_text + f"\n\n**[API error: {e}]**"
        yield chatbot, claude_msgs


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    tool_count = len(mcp.tools)
    mcp_badge = (
        f"Gmail connected — {tool_count} tools available"
        if tool_count
        else f"Gmail unavailable ({mcp.status}) — chatbot runs without email access"
    )

    with gr.Blocks(title="Claude + Gmail", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Claude + Gmail Assistant")
        gr.Markdown(
            f"**Model:** `{MODEL}` &nbsp;|&nbsp; **MCP:** {mcp_badge}"
        )

        claude_state = gr.State([])

        chatbot = gr.Chatbot(
            type="messages",
            height=560,
            show_copy_button=True,
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

        # Wire up events
        submit_event = (
            msg_box.submit(
                respond,
                inputs=[msg_box, chatbot, claude_state],
                outputs=[chatbot, claude_state],
            )
            .then(lambda: gr.update(value=""), outputs=msg_box)
        )

        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, claude_state],
            outputs=[chatbot, claude_state],
        ).then(lambda: gr.update(value=""), outputs=msg_box)

        clear_btn.click(
            lambda: ([], []),
            outputs=[chatbot, claude_state],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    print("Connecting to Gmail MCP server…")
    mcp = MCPClient(SERVER_SCRIPT)
    ok = mcp.start()
    if ok:
        print(f"Gmail MCP ready — {mcp.status}")
    else:
        print(f"Gmail MCP unavailable ({mcp.status}) — continuing without email tools")

    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
