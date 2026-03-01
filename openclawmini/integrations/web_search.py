"""
Web search and public profile scraping for OpenClawMini.

Sources:
  - DuckDuckGo search (no API key needed via duckduckgo-search library)
  - LinkedIn public profile scraping via requests + BeautifulSoup
  - General web page scraping for public mentions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    page_text: str = ""


@dataclass
class WebSearchResult:
    results: list[SearchResult] = field(default_factory=list)
    error: Optional[str] = None


def search_duckduckgo(query: str, max_results: int = 5) -> WebSearchResult:
    """
    Search DuckDuckGo and return results with URL, title, and snippet.

    Requires: pip install duckduckgo-search
    """
    result = WebSearchResult()
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        for r in raw:
            result.results.append(SearchResult(
                url=r.get("href", ""),
                title=r.get("title", ""),
                snippet=r.get("body", ""),
            ))
    except ImportError:
        result.error = "duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        result.error = str(e)
    return result


def scrape_page(url: str, timeout: int = 10) -> str:
    """
    Fetch a URL and return cleaned plain text using BeautifulSoup.

    Returns empty string on any error (network, parse, etc.).
    """
    if not url or not url.startswith("http"):
        return ""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text)
        return text[:5000]
    except Exception:
        return ""


def scrape_linkedin_public(url: str) -> tuple[str, str]:
    """
    Attempt to scrape a public LinkedIn profile URL.

    Returns (page_text, error_message).
    LinkedIn frequently blocks automated requests; we use a realistic
    User-Agent and fall back gracefully.
    """
    if not url or "linkedin.com" not in url:
        return "", "Not a LinkedIn URL"

    text = scrape_page(url)
    if not text:
        return "", (
            "Could not fetch LinkedIn page — LinkedIn may be blocking automated access. "
            "Try pasting your public profile text manually as a file upload instead."
        )
    return text, ""


def build_search_queries(name: str, email: str = "", linkedin_url: str = "") -> list[str]:
    """
    Generate web search queries to find public information about a person.

    Returns a prioritized list of query strings.
    """
    queries: list[str] = []
    if not name:
        return queries

    # LinkedIn-specific query first (highest quality for professional facts)
    queries.append(f'"{name}" site:linkedin.com')

    # General professional background
    queries.append(f'"{name}" professional background career')

    # Company/role signals
    if email:
        domain = email.split("@")[-1] if "@" in email else ""
        # Skip generic email providers
        if domain and domain not in {
            "gmail.com", "yahoo.com", "hotmail.com",
            "outlook.com", "icloud.com", "protonmail.com",
        }:
            queries.append(f'"{name}" "{domain}"')

    return queries
