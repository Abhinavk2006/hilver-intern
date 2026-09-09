import re
import html

def clean_text(text: str) -> str:
    """
    Cleans customer tweet text while preserving essential intent signals.
    - Decodes HTML entities (&amp; -> &)
    - Replaces anonymized customer user handles (@115712) with a generic tag or strips them
    - Retains target brand handle for context
    - Retains punctuation, URLs, and numbers
    - Normalizes extra whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Decode HTML entities
    cleaned = html.unescape(text)

    # Replace anonymized numeric user handles (e.g. @115712, @115858)
    cleaned = re.sub(r'@[0-9]+', '@user', cleaned)

    # Normalize multiple whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def extract_urls(text: str) -> list[str]:
    """Extracts URLs from text."""
    url_pattern = r'https?://\S+'
    return re.findall(url_pattern, text)
