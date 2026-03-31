# Claude MCP Chatbot

A multi-provider AI chatbot with Gmail integration via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). Available as both a terminal CLI and a Gradio web UI.

## Features

- **Multi-provider** — route requests through [OpenRouter](https://openrouter.ai/) (Anthropic, OpenAI, xAI) or call [xAI](https://x.ai/) directly
- **Streaming responses** with real-time token output
- **Gmail tools** — search, read, list labels, and fetch full threads via OAuth2
- **Web UI** — Gradio-based chat interface with markdown rendering
- **CLI** — lightweight terminal chatbot with conversation history save/load

---

## Project Structure

```
chatbot/
├── chatbot.py            # Terminal CLI chatbot
├── web_chatbot.py        # Gradio web UI chatbot
├── gmail_mcp_server.py   # Gmail MCP server (stdio)
└── requirements.txt
```

---

## Supported Providers & Models

| Provider | `--provider` | API Key env var | Default model |
|---|---|---|---|
| OpenRouter | `openrouter` (default) | `OPENROUTER_API_KEY` | `anthropic/claude-opus-4.6` |
| xAI | `xai` | `XAI_API_KEY` | `grok-2` |

### Model aliases (OpenRouter)

| Alias | Resolves to |
|---|---|
| `claude` | `anthropic/claude-opus-4.6` |
| `sonnet` | `anthropic/claude-sonnet-4.6` |
| `gpt4` | `openai/gpt-4o` |
| `grok` | `x-ai/grok-4.20-beta` |

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
# OpenRouter (access Anthropic, OpenAI, xAI via one key)
export OPENROUTER_API_KEY=sk-or-...

# OR xAI directly
export XAI_API_KEY=xai-...
```

### 3. Gmail OAuth setup (optional)

The Gmail integration uses Google's OAuth2. To enable it:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services → Credentials**
2. Create an **OAuth 2.0 Client ID** (Desktop app) and download `credentials.json` into this directory
3. Enable the **Gmail API** for your project
4. Add your Gmail address as a test user under **OAuth consent screen → Test users**
5. Run the MCP server once to complete the browser OAuth flow:
   ```bash
   python gmail_mcp_server.py
   ```
   A browser window opens, you approve access, and `token.json` is saved for future runs.

> If you see "Access blocked: app has not completed Google verification", click **Advanced → Go to app (unsafe)** — this is expected for personal/test OAuth apps.

---

## Usage

### Terminal CLI (`chatbot.py`)

```bash
# OpenRouter — Claude Opus 4.6 (default)
python chatbot.py

# OpenRouter — specific model alias
python chatbot.py --model sonnet
python chatbot.py --model gpt4
python chatbot.py --model grok

# OpenRouter — full model ID
python chatbot.py --model anthropic/claude-opus-4.6

# xAI directly
python chatbot.py --provider xai
python chatbot.py --provider xai --model grok-2
```

**CLI commands:**

| Command | Description |
|---|---|
| `/clear` | Clear conversation history |
| `/save` | Save history to `chat_history.json` |
| `/load` | Load history from `chat_history.json` |
| `/model` | Show current model |
| `/system` | Show current system prompt |
| `/help` | Show command list |
| `/quit` | Exit |

### Web UI (`web_chatbot.py`)

```bash
# OpenRouter — Claude Opus 4.6 (default)
python web_chatbot.py

# xAI directly
python web_chatbot.py --provider xai

# Specific model
python web_chatbot.py --model sonnet
python web_chatbot.py --provider xai --model grok-2
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Gmail MCP Tools

The MCP server exposes four read-only Gmail tools to the AI:

| Tool | Description |
|---|---|
| `gmail_search_messages` | Search with Gmail syntax, e.g. `from:alice is:unread` |
| `gmail_read_message` | Read the full body of a message by ID |
| `gmail_list_labels` | List all labels and folders |
| `gmail_get_thread` | Fetch every message in a thread (full conversation) |

The chatbot automatically uses these tools when you ask about your email — no special syntax needed.

**Examples:**
- *"Do I have any unread emails from Alice?"*
- *"Summarize my last 5 emails"*
- *"Show me the full thread about the project proposal"*

---

## Custom System Prompt

Override the default system prompt via environment variable:

```bash
export CHATBOT_SYSTEM_PROMPT="You are a concise assistant focused on software engineering."
python web_chatbot.py
```

---

## Requirements

- Python 3.10+
- `openai` — API client (used for OpenRouter and xAI)
- `gradio` 6+ — web UI
- `mcp` — Model Context Protocol client/server
- `google-api-python-client`, `google-auth-oauthlib` — Gmail API
