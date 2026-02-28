from enum import Enum


class ContentType(Enum):
    TEXT = "text"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    MIXED = "mixed"


def classify_page(analysis_text: str) -> ContentType:
    lower = analysis_text.lower()
    has_diagram = any(w in lower for w in ("diagram", "graph", "figure", "chart"))
    has_formula = any(w in lower for w in ("equation", "formula", "latex"))
    has_text = any(w in lower for w in ("text", "paragraph"))

    if has_diagram and not has_formula and not has_text:
        return ContentType.DIAGRAM
    if has_formula and not has_diagram and not has_text:
        return ContentType.FORMULA
    if has_text and not has_diagram and not has_formula:
        return ContentType.TEXT
    return ContentType.MIXED
