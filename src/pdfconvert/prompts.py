CLASSIFY_PROMPT = """Analyze this page image and classify its content.
Reply with ONLY one or more of these labels, comma-separated:
- TEXT: contains paragraphs of readable text
- FORMULA: contains mathematical or physics equations/formulas
- DIAGRAM: contains charts, graphs, figures, or diagrams

Example reply: TEXT, FORMULA"""

TEXT_EXTRACT_PROMPT = """You are a precise document OCR assistant.
Extract ALL text content from this page image exactly as it appears.
Rules:
- Preserve the original paragraph structure and line breaks.
- Keep headings, bullet points, and numbered lists in Markdown format.
- If there are tables, convert them to Markdown table syntax.
- Do NOT add any commentary or explanation. Output only the extracted text.
- For any text you cannot read clearly, mark it as [unclear].
- Preserve both English and Chinese text as-is."""

FORMULA_EXTRACT_PROMPT = """You are a LaTeX transcription assistant.
Convert ALL mathematical and physics formulas in this image to LaTeX.
Rules:
- Wrap inline formulas with single dollar signs: $...$
- Wrap display/block formulas with double dollar signs: $$...$$
- Preserve surrounding text context in plain text.
- Use standard LaTeX packages (amsmath, amssymb) notation.
- For vectors use \\vec{}, for matrices use \\begin{pmatrix}.
- If a formula is unclear, provide your best interpretation with a comment: %% [uncertain]
- Output the formulas in the order they appear on the page."""

DIAGRAM_DESCRIBE_PROMPT = """You are a technical diagram analyst.
Describe the diagram/chart/figure in this image for a Markdown document.
Rules:
- Start with a brief one-line caption: **Figure: [description]**
- Describe the diagram structure, labels, axes, and data trends.
- If it contains numerical data, extract key values into a Markdown table.
- If it is a circuit diagram or physics diagram, describe components and their connections.
- Keep the description concise but complete enough to understand without seeing the image."""

MIXED_EXTRACT_PROMPT = """You are a document analysis assistant.
This page contains mixed content (text, formulas, and/or diagrams).
Extract and convert ALL content to Markdown:
- Plain text: preserve as-is with Markdown formatting
- Mathematical formulas: convert to LaTeX ($...$ for inline, $$...$$ for block)
- Diagrams/figures: describe with **Figure: [description]** followed by a brief description
- Tables: convert to Markdown table syntax
Output only the converted content, no commentary."""
