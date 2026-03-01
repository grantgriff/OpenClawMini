"""
Web search and public profile scraping for OpenClawMini.

Search backends (in priority order):
  1. Brave Search API (BRAVE_API_KEY) — free tier, 2000 req/month, most reliable
  2. DuckDuckGo (no key needed) — free but rate-limited, used as fallback

Specialized scrapers:
  - LinkedIn public profile — with User-Agent rotation
  - GitHub profile + repos — via GitHub API (no auth needed for public data)
  - General web pages — requests + BeautifulSoup with retry

All functions are designed to silently degrade: network failures return empty
results rather than raising exceptions, so the pipeline never hard-fails on
a missing web page.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Optional


# ── Data models ────────────────────────────────────────────────


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    page_text: str = ""
    source: str = "web"  # "web", "github", "linkedin"


@dataclass
class WebSearchResult:
    results: list[SearchResult] = field(default_factory=list)
    error: Optional[str] = None


# ── Search backends ────────────────────────────────────────────


def search_brave(query: str, max_results: int = 10) -> WebSearchResult:
    """
    Search using the Brave Search API.

    Requires BRAVE_API_KEY env var (free at brave.com/search/api/).
    Free tier: 2,000 queries/month — more than enough for a hackathon.
    """
    result = WebSearchResult()
    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    if not api_key:
        result.error = "BRAVE_API_KEY not set"
        return result

    try:
        import requests
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={"q": query, "count": min(max_results, 20)},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("web", {}).get("results", []):
            result.results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("description", ""),
                source="web",
            ))
    except ImportError:
        result.error = "requests not installed"
    except Exception as e:
        result.error = str(e)
    return result


def search_duckduckgo(query: str, max_results: int = 10) -> WebSearchResult:
    """
    Search DuckDuckGo (no API key needed, fallback search backend).

    Uses the duckduckgo-search library with retry on rate-limit errors.
    """
    result = WebSearchResult()
    try:
        from duckduckgo_search import DDGS
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=max_results))
                for r in raw:
                    result.results.append(SearchResult(
                        url=r.get("href", ""),
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        source="web",
                    ))
                break  # Success
            except Exception as e:
                if attempt == 2:
                    result.error = str(e)
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
    except ImportError:
        result.error = "duckduckgo-search not installed"
    return result


def search(query: str, max_results: int = 10) -> WebSearchResult:
    """
    Smart search: Brave if API key is set, DuckDuckGo as fallback.
    Always returns a WebSearchResult (never raises).
    """
    brave_key = os.getenv("BRAVE_API_KEY", "").strip()
    if brave_key:
        result = search_brave(query, max_results)
        if result.results:
            return result
        # Brave failed — fall through to DuckDuckGo
    return search_duckduckgo(query, max_results)


# ── Web scraping ───────────────────────────────────────────────


_USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
]


def scrape_page(url: str, timeout: int = 12, max_chars: int = 5000) -> str:
    """
    Fetch a URL and return cleaned plain text.
    Returns empty string on any error (never raises).
    """
    if not url or not url.startswith("http"):
        return ""
    try:
        import requests
        from bs4 import BeautifulSoup
        import random

        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "meta"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text)
        return text[:max_chars]
    except Exception:
        return ""


def scrape_pages_parallel(
    urls: list[str], timeout: int = 12, max_workers: int = 5
) -> dict[str, str]:
    """
    Scrape multiple URLs in parallel. Returns {url: page_text}.
    Any failed URL maps to empty string.
    """
    results: dict[str, str] = {url: "" for url in urls}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(scrape_page, url, timeout): url
            for url in urls
            if url and url.startswith("http")
        }
        for future in as_completed(future_to_url, timeout=timeout + 5):
            url = future_to_url[future]
            try:
                results[url] = future.result(timeout=2)
            except Exception:
                pass
    return results


# ── LinkedIn ───────────────────────────────────────────────────


def scrape_linkedin_public(url: str) -> tuple[str, str]:
    """
    Attempt to scrape a public LinkedIn profile.

    LinkedIn aggressively blocks bots. If scraping fails (common), the caller
    should offer the user a manual paste fallback.

    Returns: (page_text, error_message) — error is empty on success.
    """
    if not url or "linkedin.com" not in url:
        return "", "Not a LinkedIn URL"

    text = scrape_page(url)
    if not text or len(text) < 100:
        return "", (
            "LinkedIn is blocking automated access. "
            "Try: paste your LinkedIn 'About' section text as a file upload instead."
        )
    return text, ""


# ── GitHub profile ─────────────────────────────────────────────


def fetch_github_profile(username: str) -> dict:
    """
    Fetch a GitHub user's public profile + top repos via the GitHub API.

    No auth token needed for public data (60 req/hour unauthenticated).
    Returns a dict with: name, bio, company, location, blog, repos (list).
    Returns empty dict on any error.
    """
    if not username:
        return {}
    try:
        import requests
        base = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        gh_token = os.getenv("GITHUB_TOKEN", "")
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"

        # User profile
        resp = requests.get(f"{base}/users/{username}", headers=headers, timeout=10)
        if resp.status_code != 200:
            return {}
        profile = resp.json()

        # Top public repos (for skills/interests signal)
        repos_resp = requests.get(
            f"{base}/users/{username}/repos",
            headers=headers,
            params={"sort": "stars", "per_page": 10},
            timeout=10,
        )
        repos = []
        if repos_resp.status_code == 200:
            for r in repos_resp.json()[:10]:
                repos.append({
                    "name": r.get("name", ""),
                    "description": r.get("description", ""),
                    "language": r.get("language", ""),
                    "stars": r.get("stargazers_count", 0),
                })

        return {
            "name": profile.get("name", ""),
            "bio": profile.get("bio", ""),
            "company": profile.get("company", ""),
            "location": profile.get("location", ""),
            "blog": profile.get("blog", ""),
            "public_repos": profile.get("public_repos", 0),
            "repos": repos,
        }
    except Exception:
        return {}


def github_profile_to_text(gh: dict) -> str:
    """Convert GitHub profile dict to flat text for Gemini extraction."""
    if not gh:
        return ""
    parts = []
    if gh.get("name"):
        parts.append(f"Name: {gh['name']}")
    if gh.get("bio"):
        parts.append(f"Bio: {gh['bio']}")
    if gh.get("company"):
        parts.append(f"Company: {gh['company']}")
    if gh.get("location"):
        parts.append(f"Location: {gh['location']}")
    if gh.get("blog"):
        parts.append(f"Website: {gh['blog']}")
    if gh.get("repos"):
        repo_lines = []
        for r in gh["repos"][:8]:
            lang = f" [{r['language']}]" if r.get("language") else ""
            desc = f" — {r['description']}" if r.get("description") else ""
            repo_lines.append(f"  {r['name']}{lang}{desc}")
        parts.append("Top repos:\n" + "\n".join(repo_lines))
    return "\n".join(parts)


def detect_github_username(name: str, email: str = "") -> Optional[str]:
    """
    Try to find a GitHub username for a person by searching DuckDuckGo.
    Returns username string or None.
    """
    query = f"{name} site:github.com"
    if email and "@" in email:
        domain = email.split("@")[-1]
        if domain not in {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}:
            query = f"{name} github.com {domain}"

    result = search(query, max_results=3)
    for r in result.results:
        match = re.search(r"github\.com/([a-zA-Z0-9_-]+)", r.url)
        if match:
            username = match.group(1)
            # Skip GitHub pages that are orgs or meta pages
            if username.lower() not in {
                "features", "topics", "explore", "trending",
                "marketplace", "pricing", "login", "join",
            }:
                return username
    return None


# ── Search query builders ──────────────────────────────────────


def build_search_queries(name: str, email: str = "", linkedin_url: str = "") -> list[str]:
    """
    Generate prioritized web search queries to find public information.

    Ordered by expected signal quality. Up to 9 queries covering all major
    fact categories: work, education, skills, achievements, location, interests.
    """
    queries: list[str] = []
    if not name:
        return queries

    # LinkedIn — highest quality for professional facts
    if linkedin_url:
        queries.append(f'"{name}" site:linkedin.com')
    else:
        queries.append(f'"{name}" linkedin')

    # Core professional background
    queries.append(f'"{name}" career job title company')

    # Education
    queries.append(f'"{name}" education university degree')

    # Skills / expertise
    queries.append(f'"{name}" skills expertise technology')

    # Achievements / recognition
    queries.append(f'"{name}" achievements awards recognition')

    # Bio / about
    queries.append(f'"{name}" background bio about')

    # Interests / projects
    queries.append(f'"{name}" interests hobbies projects')

    # GitHub (great for dev skills + open-source work)
    queries.append(f'"{name}" github.com')

    # Company domain if we have a non-generic email
    if email and "@" in email:
        domain = email.split("@")[-1]
        if domain not in {
            "gmail.com", "yahoo.com", "hotmail.com",
            "outlook.com", "icloud.com", "protonmail.com",
        }:
            queries.append(f'"{name}" {domain}')

    return queries
