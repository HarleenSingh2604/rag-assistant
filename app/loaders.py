"""Turn an uploaded file into plain text."""
import io
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".pdf"}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # corrupt or encrypted PDF
            raise ValueError("Could not read this PDF") from exc
    raise ValueError("Unsupported file type. Use .txt, .md or .pdf")
