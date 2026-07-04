import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_all_docs(doc_dir: str = "doc") -> tuple[str, list[str]]:
    doc_path = Path(doc_dir)
    full_text = ""
    all_chunks = []

    for pdf_file in sorted(doc_path.glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf_file))
        full_text += f"\n\n=== {pdf_file.name} ===\n\n{text}"
        all_chunks.extend(chunk_text(text))

    return full_text, all_chunks
