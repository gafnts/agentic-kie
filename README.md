# Agentic KIE: LLM-Based Key Information Extraction from Documents

[![CI](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml/badge.svg)](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml)
[![CD](https://github.com/gafnts/agentic-kie/actions/workflows/cd.yml/badge.svg)](https://github.com/gafnts/agentic-kie/actions/workflows/cd.yml)
[![codecov](https://codecov.io/github/gafnts/agentic-kie/graph/badge.svg)](https://codecov.io/github/gafnts/agentic-kie)
[![PyPI](https://img.shields.io/pypi/v/agentic-kie)](https://pypi.org/project/agentic-kie/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Python package for extracting structured information from PDF documents using large language models.

**agentic-kie** handles the full extraction pipeline: it loads PDFs (including scanned documents via a pluggable OCR backend), and exposes both the raw text and rendered page images so that LLMs can reason over document content using text, vision, or a combination of both. Two extraction strategies are available — a fast single-pass approach and a more capable agentic loop — designed for use in production pipelines and research workflows alike.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
  - [Loading a PDF](#loading-a-pdf)
  - [Scanned documents and OCR](#scanned-documents-and-ocr)
  - [Error handling](#error-handling)
- [Extraction strategies](#extraction-strategies)
  - [Single-pass extraction](#single-pass-extraction)
  - [Agentic extraction](#agentic-extraction)
- [Contributing](#contributing)

---

## Installation

Requires Python 3.13 or later.

```bash
pip install agentic-kie
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add agentic-kie
```

---

## Quick start

### Loading a PDF

`PDFLoader` is the main entry point. It handles file I/O, detects whether the document has a native text layer, and returns an immutable `PDFDocument` ready for downstream use.

```python
from pathlib import Path
from agentic_kie import PDFLoader

loader = PDFLoader()
doc = loader.load(Path("invoice.pdf"))

# Access the full document text
print(doc.full_text)

# Navigate by page (zero-indexed, half-open ranges)
print(doc.read_text(0, 3))   # pages 0, 1, 2
print(doc.read_text(4))      # page 4 only

# Render pages to base64-encoded PNG strings (for vision models)
images = doc.all_images          # all pages
first_page = doc.load_images(0)  # single page
```

`PDFDocument` exposes:

| Attribute / Method | Description |
|---|---|
| `page_count` | Total number of pages |
| `is_ocr` | `True` if text was extracted via OCR |
| `full_text` | All pages concatenated with double newlines |
| `read_text(start, end=None)` | Text slice over a page range |
| `all_images` | All pages as base64 PNGs (cached) |
| `load_images(start, end=None)` | Image slice over a page range |

### Scanned documents and OCR

For scanned PDFs, `PDFLoader` automatically detects the absence of a text layer and routes to an OCR provider. Any object implementing `extract_text(image: bytes) -> str` qualifies — no subclassing required.

```python
from agentic_kie import PDFLoader, OCRProvider

class TextractProvider:
    def extract_text(self, image: bytes) -> str:
        # call AWS Textract (or any OCR service)
        ...

loader = PDFLoader(ocr_provider=TextractProvider())
doc = loader.load(Path("scanned_form.pdf"))

print(doc.is_ocr)    # True
print(doc.full_text)
```

The `dpi` and `text_threshold` parameters let you control rendering resolution and the sensitivity of the native-text detection heuristic:

```python
loader = PDFLoader(
    ocr_provider=TextractProvider(),
    dpi=300,            # higher DPI improves OCR accuracy on dense documents
    text_threshold=50,  # minimum avg characters/page to skip OCR
)
```

### Error handling

All document-level failures raise from a common `DocumentLoadError` base, making them easy to catch together or individually:

```python
from agentic_kie import (
    DocumentLoadError,
    CorruptDocumentError,
    PasswordProtectedError,
    EmptyDocumentError,
    OCRNotConfiguredError,
)

try:
    doc = loader.load(path)
except PasswordProtectedError:
    print("Document is encrypted")
except OCRNotConfiguredError:
    print("Scanned document detected — provide an OCR provider")
except DocumentLoadError as e:
    print(f"Load failed: {e}")
```

---

## Extraction strategies

All extractors satisfy the `Extractor` protocol — a single `extract(document) -> T` method that takes a `PDFDocument` and returns a validated instance of your Pydantic schema. This lets you swap strategies without changing calling code.

### Single-pass extraction

`SinglePassExtractor` issues one structured LLM call and parses the response directly against a Pydantic schema. Fast, predictable, and suitable for well-structured documents.

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from agentic_kie import PDFLoader, SinglePassExtractor

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    due_date: str | None

loader = PDFLoader()
doc = loader.load(Path("invoice.pdf"))

extractor = SinglePassExtractor(
    model=ChatOpenAI(model="gpt-4o"),
    schema=Invoice,
)

result = extractor.extract(doc)
print(result.vendor, result.total)
```

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `BaseChatModel` | *required* | Any [LangChain chat model](https://python.langchain.com/docs/integrations/chat/) (ChatOpenAI, ChatAnthropic, ChatBedrock, etc.) |
| `schema` | `type[T]` | *required* | Pydantic model class defining the fields to extract |
| `modality` | `"text" \| "image" \| "multimodal"` | `"text"` | Which document representations to send to the model |
| `system_prompt` | `str \| None` | `None` | Custom system prompt (uses a sensible default when omitted) |
| `max_retries` | `int` | `3` | Maximum retry attempts on transient failures (rate limits, timeouts). Uses exponential backoff with jitter |

#### Modalities

- **`"text"`** — sends only the extracted text. Fastest and cheapest; works well when the document has a reliable text layer.
- **`"image"`** — sends rendered page images. Useful for visually rich documents where layout matters.
- **`"multimodal"`** — sends text followed by page images, giving the model both signals.

### Agentic extraction

A [LangChain](https://python.langchain.com/)-powered agent loop that can reason iteratively, call tools, and refine its output over multiple steps. Better suited for complex or ambiguous documents. *Coming soon.*

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, available `make` targets, and the CI/CD pipeline.
