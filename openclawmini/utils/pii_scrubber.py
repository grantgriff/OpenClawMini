"""
PII (Personally Identifiable Information) scrubber for OpenClawMini.

Applied to email bodies before they are stored as writing samples or sent
to Gemini for fact extraction.  We want the model to learn writing *style*
and career *facts*, not your home address or Social Security number.

What gets redacted (replaced with a labeled placeholder):
  [PHONE]   — phone numbers in US/international formats
  [SSN]     — Social Security numbers (XXX-XX-XXXX)
  [CC]      — credit/debit card numbers (16-digit blocks)
  [ADDRESS] — US street addresses (number + street-type keyword)
  [IP]      — IPv4 addresses

What is NOT redacted (we want this for fact/style learning):
  - Names of people (needed for relationship/fact extraction)
  - Company names, job titles, skills, locations at city/state level
  - Email addresses of recipients (sender context is useful)
  - URLs and domains (useful for job/project facts)
  - Standalone 5-digit numbers (years, amounts, IDs — too many false positives)
  - 9-digit numbers unless formatted as SSN (routing# regex has high false-positive rate)
"""

from __future__ import annotations

import re


# ── Regex patterns ─────────────────────────────────────────────

# US & international phone numbers, many common formats:
#   +1-800-555-0100 | (800) 555-0100 | 800.555.0100 | 8005550100
# Requires a full 10-digit number to avoid matching years/IDs.
_PHONE = re.compile(
    r"""
    (?:
        (?:\+?1[-.\s]?)?                # optional country code
        (?:\(?\d{3}\)?[-.\s]?)          # area code (3 digits)
        \d{3}[-.\s]?\d{4}              # 7-digit local number
    )
    """,
    re.VERBOSE,
)

# SSN — only the hyphenated format (XXX-XX-XXXX) to avoid false positives on
# bare 9-digit numbers (order numbers, employee IDs, etc.)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit / debit card: 16 digits grouped 4-4-4-4 with optional separators.
# Bare 16-digit runs are rare in normal email prose.
_CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b")

# US street address: house number + 1-4 words + mandatory street-type keyword.
# The street-type anchor keeps false-positive rate low.
_STREET_ADDRESS = re.compile(
    r"""
    \b\d{1,6}                   # house number
    (?:\s+\w+){1,4}             # 1-4 words (street name words)
    \s+
    (?:
        Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr
        |Lane|Ln|Court|Ct|Place|Pl|Circle|Cir|Terrace|Ter
        |Highway|Hwy|Parkway|Pkwy|Trail|Trl|Square|Sq|Loop|Way
    )
    (?:\.|\b)
    (?:\s+(?:Apt|Suite|Ste|Unit|#)\s*\w+)?   # optional unit/apt
    """,
    re.VERBOSE | re.IGNORECASE,
)

# IPv4 addresses (very distinctive format, near-zero false positives)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# ── Public API ─────────────────────────────────────────────────

def redact_pii(text: str) -> str:
    """
    Redact sensitive PII from text and return the scrubbed version.

    Replacements use clearly labeled tokens so the text remains readable
    and the model can learn natural writing flow without real PII.

    Args:
        text: Raw email body or document text.

    Returns:
        Text with PII replaced by labeled placeholders.
    """
    if not text:
        return text

    # Order matters: more-specific patterns first to avoid double-replacement.

    # SSN (hyphen-formatted) before phone — both contain digit groups
    text = _SSN.sub("[SSN]", text)

    # Credit cards — 16 digits grouped 4-4-4-4
    text = _CREDIT_CARD.sub("[CC]", text)

    # Street addresses — before phone to catch house numbers first
    text = _STREET_ADDRESS.sub("[ADDRESS]", text)

    # Phone numbers
    text = _PHONE.sub("[PHONE]", text)

    # IPv4 addresses
    text = _IPV4.sub("[IP]", text)

    return text


def has_pii(text: str) -> bool:
    """Return True if the text contains any detectable PII."""
    return bool(
        _SSN.search(text)
        or _CREDIT_CARD.search(text)
        or _PHONE.search(text)
        or _STREET_ADDRESS.search(text)
    )
