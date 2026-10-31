"""
Text cleaner: normalizes raw PDF extraction output before chunking.

Fixes the classic pypdf artifacts: hyphenated line breaks, stray control
characters, non-breaking spaces, and irregular whitespace.
"""

import re
import unicodedata

# Control characters other than newline/tab
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Word split across a line break: "engineer-\ning" -> "engineering"
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\s*\n\s*([a-z])")
# Any run of whitespace (including newlines) -> single space
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return a cleaned, normalized copy of *text*."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)      # canonical unicode form
    text = text.replace("\u00a0", " ")            # non-breaking space
    text = _HYPHEN_BREAK.sub(r"\1\2", text)        # rejoin hyphenated line breaks
    text = _CONTROL_CHARS.sub(" ", text)           # strip junk characters
    text = _WHITESPACE.sub(" ", text)              # collapse whitespace runs
    return text.strip()
