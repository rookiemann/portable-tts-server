# text_utils.py
"""Unified text processing for TTS chunking and normalization."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """NFKC normalize, straighten smart quotes, collapse whitespace, force space after periods."""
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\.(?=[^\s])', '. ', text)
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, max_chars: int = 250, min_merge: int = 30) -> list[str]:
    """Hierarchical text splitting: sentences -> clauses -> words, then merge tiny chunks.

    Splits on sentence endings (.!?), then clause boundaries (,;:-), then word
    boundaries. Never cuts mid-word. Merges trailing chunks shorter than
    ``min_merge`` characters back into the previous chunk when possible.

    Args:
        text: Input text (will be normalized first).
        max_chars: Maximum characters per chunk.
        min_merge: Chunks shorter than this get merged into the previous one.

    Returns:
        List of text chunks, each <= max_chars.
    """
    text = normalize_text(text)

    if len(text) <= max_chars:
        return [text] if text else []

    # Split into sentences on .!?
    sentences = re.split(r'(?<=[.!?])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            # Sentence too long -> split on clause boundaries
            clauses = re.split(r'(?<=[,;:\-\u2014])\s*', sentence)
            clauses = [c.strip() for c in clauses if c.strip()]
            for clause in clauses:
                if len(clause) > max_chars:
                    # Clause too long -> split on words
                    words = clause.split()
                    sub = ""
                    for word in words:
                        test = (sub + " " + word).strip() if sub else word
                        if len(test) > max_chars:
                            if sub:
                                if current and len(current + " " + sub) <= max_chars:
                                    current = (current + " " + sub).strip()
                                else:
                                    if current:
                                        chunks.append(current)
                                        current = ""
                                    chunks.append(sub)
                            sub = word
                        else:
                            sub = test
                    if sub:
                        if current and len(current + " " + sub) <= max_chars:
                            current = (current + " " + sub).strip()
                        else:
                            if current:
                                chunks.append(current)
                                current = ""
                            current = sub
                else:
                    test = (current + " " + clause).strip() if current else clause
                    if len(test) > max_chars:
                        if current:
                            chunks.append(current)
                            current = ""
                        current = clause
                    else:
                        current = test
        else:
            test = (current + " " + sentence).strip() if current else sentence
            if len(test) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                current = sentence
            else:
                current = test

    if current:
        chunks.append(current)

    # Merge tiny trailing chunks
    i = 0
    while i < len(chunks) - 1:
        if len(chunks[i + 1]) < min_merge:
            test = (chunks[i] + " " + chunks[i + 1]).strip()
            if len(test) <= max_chars:
                chunks[i] = test
                del chunks[i + 1]
                continue
        i += 1

    return chunks


def chunk_text_for_model(text: str, model_id: str) -> list[str]:
    """Convenience wrapper using per-model character limits.

    Args:
        text: Input text.
        model_id: One of 'xtts', 'fish', 'kokoro', etc.

    Returns:
        List of text chunks sized for the given model.
    """
    from audio_profiles import TEXT_LIMITS
    max_chars = TEXT_LIMITS.get(model_id, TEXT_LIMITS["default"])
    min_merge = 40 if model_id == "kokoro" else 30
    return chunk_text(text, max_chars=max_chars, min_merge=min_merge)


def sanitize_for_whisper(text: str) -> str:
    """Lowercase, strip punctuation for fuzzy Whisper comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()
