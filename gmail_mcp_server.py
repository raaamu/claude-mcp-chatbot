#!/usr/bin/env python3
"""
Gmail MCP server — exposes Gmail read-only tools over stdio.

First-time setup:
  1. Go to https://console.cloud.google.com → APIs & Services → Credentials
  2. Create an OAuth 2.0 Client ID (Desktop app) and download credentials.json
     into this directory.
  3. Enable the Gmail API for your project.
  4. Run once manually to complete the browser OAuth flow:
       python gmail_mcp_server.py
     A browser window opens, you approve access, and token.json is created.
  Subsequent runs (and the web chatbot) reuse token.json silently.
"""

import base64
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BASE_DIR = Path(__file__).parent
CREDS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"

mcp = FastMCP("gmail")


# ── Auth ─────────────────────────────────────────────────────────────────────

def _get_service():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_PATH.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDS_PATH}.\n"
                    "Download it from Google Cloud Console (OAuth 2.0 Desktop app)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers_dict(payload: dict) -> dict:
    return {h["name"]: h["value"] for h in payload.get("headers", [])}


def _extract_text_body(part: dict) -> str:
    """Recursively extract plain-text body from a message part."""
    if part.get("mimeType") == "text/plain":
        data = part.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    for sub in part.get("parts", []):
        result = _extract_text_body(sub)
        if result:
            return result
    return ""


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def gmail_search_messages(query: str, max_results: int = 10) -> str:
    """
    Search Gmail using standard Gmail search syntax.
    Examples: 'from:alice is:unread', 'subject:invoice after:2024/1/1'.
    Returns a summary list with message IDs, senders, subjects, and dates.
    """
    service = _get_service()
    res = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()

    messages = res.get("messages", [])
    if not messages:
        return "No messages found."

    summaries = []
    for msg in messages:
        detail = service.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        h = _headers_dict(detail.get("payload", {}))
        snippet = detail.get("snippet", "")[:120]
        summaries.append(
            f"ID: {msg['id']}\n"
            f"From: {h.get('From', 'Unknown')}\n"
            f"Subject: {h.get('Subject', '(no subject)')}\n"
            f"Date: {h.get('Date', 'Unknown')}\n"
            f"Snippet: {snippet}"
        )

    return "\n\n---\n\n".join(summaries)


@mcp.tool()
def gmail_read_message(message_id: str) -> str:
    """
    Read the full content of a Gmail message by its ID.
    Returns headers (From, To, Subject, Date) and the plain-text body.
    """
    service = _get_service()
    msg = service.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()

    payload = msg.get("payload", {})
    h = _headers_dict(payload)
    body = _extract_text_body(payload) or "[No plain-text body found]"

    return (
        f"From:    {h.get('From', 'Unknown')}\n"
        f"To:      {h.get('To', 'Unknown')}\n"
        f"Subject: {h.get('Subject', '(no subject)')}\n"
        f"Date:    {h.get('Date', 'Unknown')}\n"
        f"\n{body}"
    )


@mcp.tool()
def gmail_list_labels() -> str:
    """List all Gmail labels/folders (Inbox, Sent, custom labels, etc.)."""
    service = _get_service()
    res = service.users().labels().list(userId="me").execute()
    labels = res.get("labels", [])
    system = [l for l in labels if l.get("type") == "system"]
    user = [l for l in labels if l.get("type") == "user"]

    lines = ["=== System labels ==="]
    lines += [f"  {l['name']} (id: {l['id']})" for l in system]
    if user:
        lines += ["", "=== Custom labels ==="]
        lines += [f"  {l['name']} (id: {l['id']})" for l in user]
    return "\n".join(lines)


@mcp.tool()
def gmail_get_thread(thread_id: str) -> str:
    """
    Retrieve all messages in a Gmail thread by thread ID.
    Useful for reading a full email conversation.
    """
    service = _get_service()
    thread = service.users().threads().get(
        userId="me", id=thread_id, format="full"
    ).execute()

    parts = []
    for i, msg in enumerate(thread.get("messages", []), 1):
        payload = msg.get("payload", {})
        h = _headers_dict(payload)
        body = _extract_text_body(payload) or "[No plain-text body]"
        parts.append(
            f"--- Message {i} ---\n"
            f"From:    {h.get('From', 'Unknown')}\n"
            f"Date:    {h.get('Date', 'Unknown')}\n"
            f"\n{body.strip()}"
        )

    return "\n\n".join(parts) if parts else "No messages in thread."


if __name__ == "__main__":
    mcp.run()
