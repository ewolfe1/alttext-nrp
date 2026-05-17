import json

base_prompt = """
You are generating alt-text for digitized archival items to meet ADA accessibility
requirements (Title II, 2026).

You will be given an image and a Dublin Core metadata record (which may be empty, partial, or
inconsistent). The metadata was created over many decades and may be vague, outdated,
or occasionally inaccurate. Use it to help interpret what you are
looking at, but trust what you can actually see in the image.

The metadata is context only. It is NOT material to copy into the description: the
full record is already displayed to users alongside the image, so the alt-text does
not need to repeat any of it.

- Do NOT include names, dates, locations, or other specific identifying details,
  even when the metadata provides them. The alt-text is a general visual
  description; specific facts live in the metadata record, not the alt-text.
- Do not include URLs or identification numbers.
- Where the medium is identifiable (photograph, drawing, map, etc.), lead with it:
  e.g. "Photograph of..." or "Pencil sketch of...".
- Do not begin with "Image of", "Picture of", or "Page from".
- Write in plain descriptive language, present tense, third person.
- HARD LIMIT: the alt-text must be 150 characters or fewer. This is a ceiling, not
  a target. For a complex image, describe the most important visual elements and
  omit lesser detail rather than exceeding the limit.
- If the image is too degraded to describe reliably, say so briefly.

Respond ONLY with valid JSON, no other text:
{
  "alt_text": "...",
  "confidence": "high" | "medium" | "low",
  "notes": ""
}

The "notes" field exists ONLY to flag an item for human review. Leave it as an
empty string ("") unless one of the following is true:
- the image is too degraded to describe reliably
- the metadata clearly contradicts what is visible in the image
- you are genuinely unsure whether your description is accurate

Do NOT use "notes" to justify or explain the description, to restate the metadata,
or to record your reasoning. Most items should have an empty "notes" field.

Confidence guide:
- high: image is clear and you can describe its content with certainty
- medium: some uncertainty — image is partially degraded or content is ambiguous
- low: significant uncertainty — image is too degraded to describe reliably, or you
  are unsure whether the description is accurate
"""

book_item_prompt = """
This image is one page from a multi-page digitized object. The metadata describes
the parent object — use it for context only, do not repeat it in the alt-text.
(e.g., not "A page from Peter Rabbit" — just describe what is on this page.)

Describe the page based on its content type:

- Text-only page: describe generically by type if identifiable
  (e.g., "Title page", "Table of contents", "Blank page", "Page of printed text",
  "Page of handwritten text", "Newspaper page")
- Page containing a significant visual element (illustration, photograph, map,
  chart, diagram): describe the visual content specifically
- Mixed page (text + image): describe the visual element; note the text only if
  it is itself significant (e.g., a caption, headline, or title)
- Cover, front matter, or back matter: describe what is visible

DO NOT attempt to transcribe body text. If the page is too degraded to identify
its content type, say so.
"""

def build_prompt(metadata, image_url, item_type):
    """Returns messages list for OpenAI-compatible API"""
    if item_type == 'page':
        prompt = base_prompt + book_item_prompt
    else:
        prompt = base_prompt

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": f"Metadata:\n{json.dumps(metadata, indent=2)}"}
        ]}
    ]
