"""Split documents into chunks small enough to embed and retrieve."""
import re


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Split one very long paragraph into overlapping pieces, preferring word boundaries."""
    pieces = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return pieces


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks of at most `chunk_size` characters.

    Paragraphs longer than `chunk_size` are split with `overlap` characters shared
    between neighbouring pieces so a sentence cut at the boundary is not lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph] if len(paragraph) <= chunk_size else _hard_split(paragraph, chunk_size, overlap)
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= chunk_size:
                current += "\n\n" + piece
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks
