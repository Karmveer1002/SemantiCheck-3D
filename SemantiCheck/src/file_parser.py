import io
from pypdf import PdfReader
from docx import Document


def extract_text_from_file(uploaded_file) -> str:
    """Extracts plain text from uploaded .txt, .docx, or .pdf files."""
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if filename.endswith(".txt"):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw_bytes.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="ignore").strip()

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(raw_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    elif filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    raise ValueError(f"Unsupported file format: {filename}")
