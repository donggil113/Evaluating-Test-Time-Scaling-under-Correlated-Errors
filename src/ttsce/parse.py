"""Answer and verdict extraction for the released sc-genrm-scaling traces.

Parsing rules depend on output text only (never on the answer key) and are
applied identically to every candidate/verification. Unparseable outputs
are kept as None (never silently dropped).
"""
import re

_ANCHOR = re.compile(r"final answer is|final answer:|answer is", re.IGNORECASE)
_STRIP = re.compile(r"\\boxed\{|\\text\{|\\textbf\{|[\$\*\(\)\{\}\[\]:]")
_LETTER = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")

_VERDICT = re.compile(
    r"Is the answer correct\s*(?:\(Yes/No\)\??\s*\**\s*(Yes|No)\b|\((Yes|No)\))", re.IGNORECASE)


def extract_choice(text):
    """Return the letter A-D stated right after the last answer anchor, or None."""
    anchors = list(_ANCHOR.finditer(text))
    for a in reversed(anchors):
        snippet = _STRIP.sub(" ", text[a.end(): a.end() + 40])
        m = _LETTER.match(snippet.strip())
        if m:
            return m.group(1)
    return None


def extract_verdict(text):
    """Return 1 for a final 'Yes', 0 for a final 'No', None if absent."""
    m = _VERDICT.findall(text)
    if not m:
        return None
    word = m[-1][0] or m[-1][1]
    return 1 if word.lower() == "yes" else 0
