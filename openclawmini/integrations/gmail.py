"""
Gmail integration for OpenClawMini — read-only OAuth + email fetching.

Security:
  - Scope: gmail.readonly ONLY (cannot send, delete, or modify emails)
  - Token stored locally in ./data/gmail_token.json
  - User sees exactly what permissions are requested during OAuth
"""

from __future__ import annotations

import base64
import email as email_lib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Lazy imports so the module can be imported without the Google libs
# being installed (they're optional at import time, required at runtime).
def _google_imports():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    return Credentials, Request, InstalledAppFlow, build


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass
class ParsedEmail:
    """A parsed, cleaned email ready for memory classification."""
    message_id: str
    subject: str
    sender: str
    recipient: str
    date: str
    body: str                     # Plain text body, cleaned
    word_count: int = 0
    label: str = "SENT"          # Gmail label (SENT, INBOX, etc.)
    thread_id: str = ""

    def __post_init__(self):
        self.word_count = len(self.body.split())


@dataclass
class GmailFetchResult:
    emails: list[ParsedEmail] = field(default_factory=list)
    total_fetched: int = 0
    skipped_short: int = 0       # emails too short to be useful
    skipped_no_body: int = 0
    error: Optional[str] = None


class GmailClient:
    """
    Read-only Gmail client.

    Handles OAuth token creation/refresh and email fetching.
    Only ever requests gmail.readonly scope.
    """

    # Minimum body length to include (filter out 1-liners like "thanks!")
    MIN_BODY_WORDS = 10
    # Max results to fetch per call (Gmail API page size max is 500)
    DEFAULT_MAX_RESULTS = 500

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_path: str = "./data/gmail_token.json",
        redirect_uri: str = "http://localhost:8080/callback",
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_path = Path(token_path)
        self.redirect_uri = redirect_uri
        self._service = None

    # ── Auth ──────────────────────────────────────────────────

    def authenticate(self) -> None:
        """
        Run OAuth flow if no valid token exists, otherwise refresh.
        Opens browser for user consent on first run.
        """
        Credentials, Request, InstalledAppFlow, build = _google_imports()

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), GMAIL_SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                client_config = {
                    "installed": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "redirect_uris": [self.redirect_uri, "urn:ietf:wg:oauth:2.0:oob"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, GMAIL_SCOPES)
                creds = flow.run_local_server(port=8080, open_browser=True)

            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json())

        _, _, _, build = _google_imports()
        self._service = build("gmail", "v1", credentials=creds)

    def is_authenticated(self) -> bool:
        """Check if a valid token already exists (skips browser)."""
        if not self.token_path.exists():
            return False
        try:
            Credentials, Request, _, _ = _google_imports()
            creds = Credentials.from_authorized_user_file(str(self.token_path), GMAIL_SCOPES)
            if creds and creds.valid:
                return True
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self.token_path.write_text(creds.to_json())
                _, _, _, build = _google_imports()
                self._service = build("gmail", "v1", credentials=creds)
                return True
        except Exception:
            pass
        return False

    def _ensure_authenticated(self) -> None:
        if self._service is None:
            if self.is_authenticated():
                Credentials, _, _, build = _google_imports()
                creds = Credentials.from_authorized_user_file(str(self.token_path), GMAIL_SCOPES)
                self._service = build("gmail", "v1", credentials=creds)
            else:
                self.authenticate()

    # ── Fetching ──────────────────────────────────────────────

    def fetch_sent_emails(
        self,
        max_results: int = DEFAULT_MAX_RESULTS,
        progress_callback=None,
    ) -> GmailFetchResult:
        """
        Fetch sent emails from Gmail.

        Args:
            max_results: Max emails to retrieve (default 200).
            progress_callback: Optional callable(fetched, total) for progress updates.

        Returns:
            GmailFetchResult with parsed emails.
        """
        self._ensure_authenticated()
        result = GmailFetchResult()

        try:
            # List sent message IDs
            list_resp = (
                self._service.users()
                .messages()
                .list(userId="me", labelIds=["SENT"], maxResults=max_results)
                .execute()
            )
            messages = list_resp.get("messages", [])
            result.total_fetched = len(messages)

            for i, msg_ref in enumerate(messages):
                if progress_callback:
                    progress_callback(i + 1, len(messages))

                parsed = self._fetch_and_parse(msg_ref["id"])
                if parsed is None:
                    result.skipped_no_body += 1
                    continue
                if parsed.word_count < self.MIN_BODY_WORDS:
                    result.skipped_short += 1
                    continue
                result.emails.append(parsed)

        except Exception as e:
            result.error = str(e)

        return result

    def _fetch_and_parse(self, message_id: str) -> Optional[ParsedEmail]:
        """Fetch a single message and parse it into a ParsedEmail."""
        try:
            msg = (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except Exception:
            return None

        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        subject = headers.get("Subject", "(no subject)")
        sender = headers.get("From", "")
        recipient = headers.get("To", "")
        date = headers.get("Date", "")

        body = self._extract_body(msg.get("payload", {}))
        if not body:
            return None

        return ParsedEmail(
            message_id=message_id,
            subject=subject,
            sender=sender,
            recipient=recipient,
            date=date,
            body=body,
            thread_id=msg.get("threadId", ""),
        )

    # ── Body extraction ───────────────────────────────────────

    def _extract_body(self, payload: dict) -> str:
        """
        Recursively extract the best plain-text body from a Gmail payload.
        Prefers text/plain; falls back to stripping HTML from text/html.
        """
        mime_type = payload.get("mimeType", "")
        parts = payload.get("parts", [])

        if mime_type == "text/plain":
            return self._decode_body(payload.get("body", {}))

        if mime_type == "text/html":
            html = self._decode_body(payload.get("body", {}))
            return _strip_html(html)

        # Multipart: prefer text/plain parts
        plain_text = ""
        html_fallback = ""
        for part in parts:
            part_mime = part.get("mimeType", "")
            if part_mime == "text/plain":
                plain_text = self._decode_body(part.get("body", {}))
                if plain_text:
                    break
            elif part_mime == "text/html" and not html_fallback:
                html_raw = self._decode_body(part.get("body", {}))
                html_fallback = _strip_html(html_raw)
            elif part_mime.startswith("multipart/"):
                # Recurse into nested multipart
                nested = self._extract_body(part)
                if nested:
                    plain_text = nested
                    break

        return _clean_body(plain_text or html_fallback)

    @staticmethod
    def _decode_body(body_part: dict) -> str:
        """Base64url-decode a Gmail body part."""
        data = body_part.get("data", "")
        if not data:
            return ""
        try:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        except Exception:
            return ""


# ── Text cleaning helpers ─────────────────────────────────────

def _strip_html(html: str) -> str:
    """Strip HTML tags from a string using BeautifulSoup."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except ImportError:
        # Fallback: basic regex strip
        return re.sub(r"<[^>]+>", " ", html)


def _clean_body(text: str) -> str:
    """
    Clean email body text:
    - Remove quoted reply blocks (lines starting with >)
    - Remove excessive whitespace
    - Remove email signatures (lines after --, Best,, Regards,, etc.)
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned = []
    sig_starters = {"--", "best,", "best regards,", "regards,", "cheers,", "thanks,", "sincerely,", "sent from"}

    for line in lines:
        stripped = line.strip()
        # Stop at quoted reply blocks
        if stripped.startswith(">"):
            break
        # Stop at common signature starters
        if stripped.lower() in sig_starters or stripped.lower().startswith("sent from my"):
            break
        # Skip lines that look like forwarded message headers
        if re.match(r"^-+\s*(forwarded|original)\s+message\s*-+", stripped, re.IGNORECASE):
            break
        cleaned.append(stripped)

    result = " ".join(line for line in cleaned if line)
    # Collapse multiple spaces
    result = re.sub(r" {2,}", " ", result).strip()
    return result


# ── Factory from env/config ───────────────────────────────────

def gmail_client_from_env() -> Optional[GmailClient]:
    """
    Build a GmailClient from environment variables.
    Returns None if credentials aren't configured.
    """
    client_id = os.getenv("GMAIL_CLIENT_ID", "").strip()
    client_secret = os.getenv("GMAIL_CLIENT_SECRET", "").strip()
    token_path = os.getenv("GMAIL_TOKEN_PATH", "./data/gmail_token.json").strip()
    redirect_uri = os.getenv("GMAIL_REDIRECT_URI", "http://localhost:8080/callback").strip()

    if not client_id or not client_secret:
        return None

    return GmailClient(
        client_id=client_id,
        client_secret=client_secret,
        token_path=token_path,
        redirect_uri=redirect_uri,
    )
