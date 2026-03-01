"""
Log parsers for uploaded AI interaction exports.

Supported formats:
  - ChatGPT export (conversations.json)
  - Claude export (claude_conversations.json)
  - Generic log (list of {"role": ..., "content": ...} dicts)

Each parser returns a list of Conversation dicts ready for memory extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    role: str        # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    id: str
    title: str
    messages: list[Message] = field(default_factory=list)
    source_format: str = "generic"

    def user_messages(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "user"]

    def assistant_messages(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "assistant"]

    def as_pairs(self) -> list[tuple[str, str]]:
        """Return consecutive (user, assistant) message pairs."""
        pairs = []
        for i in range(len(self.messages) - 1):
            if self.messages[i].role == "user" and self.messages[i + 1].role == "assistant":
                pairs.append((self.messages[i].content, self.messages[i + 1].content))
        return pairs


def detect_format(data: Any) -> str:
    """Detect the log format from parsed JSON data."""
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "mapping" in first and "conversation_id" in first:
                return "chatgpt"
            if "uuid" in first and "chat_messages" in first:
                return "claude"
            if "role" in first and "content" in first:
                return "generic"
    return "unknown"


def parse_log_file(path: str) -> list[Conversation]:
    """
    Auto-detect format and parse a log file into Conversation objects.

    Raises:
        ValueError: if the file format is not recognized.
    """
    with open(path) as f:
        data = json.load(f)

    fmt = detect_format(data)
    if fmt == "chatgpt":
        return parse_chatgpt_export(data)
    elif fmt == "claude":
        return parse_claude_export(data)
    elif fmt == "generic":
        return parse_generic_log(data)
    else:
        raise ValueError(
            f"Unrecognized log format in {path}. "
            "Expected ChatGPT, Claude, or generic role/content list."
        )


def parse_chatgpt_export(data: list[dict]) -> list[Conversation]:
    """
    Parse ChatGPT's conversations.json export format.

    Structure:
      [{"conversation_id": "...", "title": "...", "mapping": {"node_id": {"message": {...}}}}]
    """
    conversations = []
    for raw in data:
        conv_id = raw.get("conversation_id") or raw.get("id", "unknown")
        title = raw.get("title", "Untitled")
        messages: list[Message] = []

        # ChatGPT uses a tree structure via "mapping"; walk it in order
        mapping = raw.get("mapping", {})
        # Build ordered list by following parent → children links
        ordered_nodes = _flatten_chatgpt_mapping(mapping)

        for node in ordered_nodes:
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role", "")
            if role not in ("user", "assistant"):
                continue
            parts = msg.get("content", {}).get("parts", [])
            content = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
            if content:
                messages.append(Message(role=role, content=content))

        if messages:
            conversations.append(Conversation(
                id=conv_id,
                title=title,
                messages=messages,
                source_format="chatgpt",
            ))

    return conversations


def _flatten_chatgpt_mapping(mapping: dict) -> list[dict]:
    """Flatten ChatGPT's tree mapping into a linear ordered list."""
    # Find root (node with no parent or parent is None)
    root_id = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            root_id = node_id
            break

    if root_id is None:
        # Fallback: just return values in insertion order
        return list(mapping.values())

    ordered = []
    visited = set()
    stack = [root_id]
    while stack:
        current_id = stack.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        node = mapping.get(current_id, {})
        ordered.append(node)
        children = node.get("children", [])
        stack = children + stack  # breadth-first

    return ordered


def parse_claude_export(data: list[dict]) -> list[Conversation]:
    """
    Parse Claude's conversation export format.

    Structure:
      [{"uuid": "...", "name": "...", "chat_messages": [{"sender": "human"/"assistant", "text": "..."}]}]
    """
    conversations = []
    for raw in data:
        conv_id = raw.get("uuid", "unknown")
        title = raw.get("name", "Untitled")
        messages: list[Message] = []

        for msg in raw.get("chat_messages", []):
            sender = msg.get("sender", "")
            role = "user" if sender == "human" else "assistant" if sender == "assistant" else None
            if not role:
                continue
            content = msg.get("text", "").strip()
            if content:
                messages.append(Message(role=role, content=content))

        if messages:
            conversations.append(Conversation(
                id=conv_id,
                title=title,
                messages=messages,
                source_format="claude",
            ))

    return conversations


def parse_generic_log(data: list[dict]) -> list[Conversation]:
    """
    Parse a generic conversation log.

    Expected structure (list of turns):
      [{"role": "user"|"assistant", "content": "..."}]

    Or a list of conversations:
      [{"messages": [{"role": "...", "content": "..."}]}]
    """
    # Single flat conversation (list of turns)
    if data and "role" in data[0]:
        messages = []
        for turn in data:
            role = turn.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = turn.get("content", "").strip()
            if content:
                messages.append(Message(role=role, content=content))
        return [Conversation(id="generic_0", title="Imported Conversation", messages=messages, source_format="generic")]

    # List of conversation objects
    conversations = []
    for i, raw in enumerate(data):
        messages = []
        for turn in raw.get("messages", []):
            role = turn.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = turn.get("content", "").strip()
            if content:
                messages.append(Message(role=role, content=content))
        if messages:
            conversations.append(Conversation(
                id=raw.get("id", f"generic_{i}"),
                title=raw.get("title", f"Conversation {i+1}"),
                messages=messages,
                source_format="generic",
            ))

    return conversations
