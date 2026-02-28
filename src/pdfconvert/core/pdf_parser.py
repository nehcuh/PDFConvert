from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from ..config import DEFAULT_DPI


def pdf_to_images(pdf_path: str | Path, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    return convert_from_path(str(pdf_path), dpi=dpi)


def pdf_page_count(pdf_path: str | Path) -> int:
    from pdf2image.pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(str(pdf_path))
    return info["Pages"]
