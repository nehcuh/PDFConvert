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
- Use [IMAGE_PLACEHOLDER: brief description | bbox: y1,y2] to mark where the diagram should be cropped
- bbox coordinates are vertical percentages (0-100) indicating the top and bottom of the region to crop
- The region should include the diagram AND any closely related labels, captions, or definitions
- Exclude unrelated text that appears before or after the diagram
- After the placeholder, provide a detailed text description of the diagram
- Describe the diagram structure, labels, axes, and data trends
- If it contains numerical data, extract key values into a Markdown table

Example output:
[IMAGE_PLACEHOLDER: Circuit diagram showing resistors R1, R2 in series with voltage source V | bbox: 20,45]

The circuit consists of two resistors connected in series..."""

MIXED_EXTRACT_PROMPT = """You are a document analysis assistant.
This page contains mixed content (text, formulas, and/or diagrams).
Extract and convert ALL content to Markdown:
- Plain text: preserve as-is with Markdown formatting
- Mathematical formulas: convert to LaTeX ($...$ for inline, $$...$$ for block)
- Diagrams/figures: use [IMAGE_PLACEHOLDER: brief description | bbox: y1,y2] to mark where each diagram should be cropped
  - bbox coordinates are VERTICAL percentages (0-100) measured from the TOP of the page
  - y1 = percentage where the diagram region STARTS (include any title/caption above the diagram)
  - y2 = percentage where the diagram region ENDS (include any formulas/labels below the diagram)
  - The cropped region should be GENEROUS - include the complete diagram with all related text
  - Example: if a diagram with labels occupies roughly the middle third of the page, use bbox: 30,70
  - If multiple diagrams exist, mark each one separately with its own bbox
- Tables: convert to Markdown table syntax
- Place [IMAGE_PLACEHOLDER] at the appropriate position in the content flow

IMPORTANT: When estimating bbox coordinates, think about the vertical position on the page:
- Top of page = 0%
- Middle of page = 50%
- Bottom of page = 100%
- A diagram in the upper portion might be bbox: 20,45
- A diagram in the middle might be bbox: 40,65
- A diagram in the lower portion might be bbox: 60,85

Example:
The concept of circular motion involves:

**Definition:** Angular displacement (θ): The angle swept by the object as it moves round.

[IMAGE_PLACEHOLDER: Circular motion diagram showing radius r, arc length s, angle θ, with formulas s=rθ, v=rω | bbox: 20,45]

The centripetal acceleration is given by: $a = v^2/r$

For a conical pendulum:

[IMAGE_PLACEHOLDER: Conical pendulum diagram with tension T, angle θ, and force equations | bbox: 55,75]

Output only the converted content, no commentary."""
