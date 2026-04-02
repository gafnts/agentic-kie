# Agentic KIE

[![CI](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml/badge.svg)](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml)
[![CD](https://github.com/gafnts/agentic-kie/actions/workflows/cd.yml/badge.svg)](https://github.com/gafnts/agentic-kie/actions/workflows/cd.yml)
[![codecov](https://codecov.io/github/gafnts/agentic-kie/graph/badge.svg)](https://codecov.io/github/gafnts/agentic-kie)
[![PyPI](https://img.shields.io/pypi/v/agentic-kie)](https://pypi.org/project/agentic-kie/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Structured key information extraction from PDF documents, powered by LLMs.

---

## The problem

Extracting structured data from PDFs is deceptively hard. The file format is a rendering instruction set, not a data container. Text layers may be missing, malformed, or absent entirely in scanned documents. Layout carries semantic meaning that raw text extraction destroys. And once you have the content, you still need an orchestration layer that let a LLM reason over it, produce typed output, and handle the inevitable failures.

## The idea

A document enters the system as a file path. It leaves as a validated Pydantic instance. Everything in between — text-layer detection, OCR routing, image rendering, LLM orchestration, output parsing, retry logic — is the library's responsibility.

Two extraction strategies are available:

- **Single-pass**: One structured LLM call. Fast, deterministic, cheap. Suitable when the document is well-structured and the target schema is straightforward.
- **Agentic**: A ReAct agent loop with document tools. The agent decides which pages to read and in what order. Suited for complex or ambiguous documents where iterative reasoning outperforms a single pass.

Both strategies satisfy the same protocol and return the same type. Swap one for the other without changing downstream code.

```python
from pathlib import Path
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from agentic_kie import PDFLoader, SinglePassExtractor, AgenticExtractor

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    due_date: str | None

model = ChatAnthropic(model="claude-haiku-4-5")
doc = PDFLoader().load(Path("invoice.pdf"))

# Fast path: single LLM call
single = SinglePassExtractor(model=model, schema=Invoice)
result = single.extract(doc)

# Or let an agent reason over the document
agent = AgenticExtractor(model=model, schema=Invoice)
result = agent.extract(doc)
```

**agentic-kie** packages that entire workflow into a typed, tested library with a clear separation of concerns: document ingestion, content representation, and structured extraction.

---

## Contents

- [Installation](#installation)
- [Core abstractions](#core-abstractions)
  - [PDFLoader](#pdfloader)
  - [PDFDocument](#pdfdocument)
  - [OCRProvider](#ocrprovider)
  - [Extractors](#extractors)
- [Extraction strategies](#extraction-strategies)
  - [Single-pass extraction](#single-pass-extraction)
  - [Agentic extraction](#agentic-extraction)
- [Modalities](#modalities)
- [Error handling](#error-handling)
- [Contributing](#contributing)

---

## Installation

Requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv add agentic-kie
```

Install with a model provider:

```bash
uv add "agentic-kie[anthropic]"   # Claude
uv add "agentic-kie[openai]"      # GPT
uv add "agentic-kie[google]"      # Gemini
uv add "agentic-kie[bedrock]"     # AWS Bedrock
uv add "agentic-kie[all]"         # All of the above
```

Any [LangChain chat model](https://python.langchain.com/docs/integrations/chat/) works. The extras above are provided for convenience.

---

## Core abstractions

The library is organized around four concepts: a loader that absorbs PDF complexity, an immutable document that exposes content, a protocol for pluggable OCR, and extractors that produce structured output.

### PDFLoader

The ingestion boundary. Takes a file path, detects whether the document has a native text layer (using a characters-per-page heuristic), routes to OCR when needed, and returns a validated `PDFDocument`.

```python
from pathlib import Path
from agentic_kie import PDFLoader

loader = PDFLoader()
doc = loader.load(Path("contract.pdf"))
```

For scanned documents, pass an OCR provider:

```python
loader = PDFLoader(
    ocr_provider=MyOCRBackend(),
    dpi=300,
    text_threshold=50,
)
doc = loader.load(Path("scanned_form.pdf"))
```

### PDFDocument

An immutable representation of the loaded document. Exposes text and rendered page images — the two modalities that LLMs can reason over. Images are rendered lazily and cached on first access.

| Attribute / Method | Description |
|---|---|
| `page_count` | Total number of pages |
| `is_ocr` | `True` if text was extracted via OCR |
| `full_text` | All pages joined with double newlines |
| `read_text(start, end=None)` | Text slice over a page range (zero-indexed, half-open) |
| `all_images` | All pages as base64-encoded PNGs (cached) |
| `load_images(start, end=None)` | Image slice over a page range |

### OCRProvider

A structural protocol. Any object with an `extract_text(image: bytes) -> str` method qualifies — no subclassing or registration required.

```python
from agentic_kie import OCRProvider

class TextractProvider:
    """Wraps AWS Textract as an OCR backend."""

    def extract_text(self, image: bytes) -> str:
        # call Textract, return plain text
        ...

# TextractProvider satisfies OCRProvider by structure alone
loader = PDFLoader(ocr_provider=TextractProvider())
```

### Extractors

Both extraction strategies satisfy the `Extractor` protocol: a single `extract(document) -> T` method that takes a `PDFDocument` and returns a validated instance of a Pydantic schema. This enables type-safe dispatch without coupling strategies through inheritance.

---

## Extraction strategies

### Single-pass extraction

`SinglePassExtractor` sends the full document content to the model in one call, with structured output bound to the target schema. The chain is built once at construction time and reused across documents.

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from agentic_kie import PDFLoader, SinglePassExtractor

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    due_date: str | None

doc = PDFLoader().load(Path("invoice.pdf"))

extractor = SinglePassExtractor(
    model=ChatOpenAI(model="gpt-5.4-mini"),
    schema=Invoice,
    modality="multimodal",
    max_retries=3,
)

result = extractor.extract(doc)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `BaseChatModel` | *required* | Any LangChain chat model |
| `schema` | `type[T]` | *required* | Pydantic model defining the extraction target |
| `modality` | `"text" \| "image" \| "multimodal"` | `"text"` | Document representation sent to the model |
| `system_prompt` | `str \| None` | `None` | Custom system prompt (uses a sensible default when omitted) |
| `max_retries` | `int` | `3` | Retry attempts with exponential backoff and jitter |

### Agentic extraction

`AgenticExtractor` builds a ReAct agent equipped with document tools — `get_page_count`, `read_text`, and `load_images` — scoped to the document being extracted. The agent decides which pages to inspect, in what order, and stops when it has enough information to produce the target schema.

```python
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from agentic_kie import PDFLoader, AgenticExtractor

class Contract(BaseModel):
    parties: list[str]
    effective_date: str
    governing_law: str | None
    termination_clause: str | None

doc = PDFLoader().load(Path("contract.pdf"))

extractor = AgenticExtractor(
    model=ChatGoogleGenerativeAI(model="gemini-3-flash"),
    schema=Contract,
    modality="text",
    max_iterations=50,
)

result = extractor.extract(doc)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `BaseChatModel` | *required* | Any LangChain chat model |
| `schema` | `type[T]` | *required* | Pydantic model defining the extraction target |
| `modality` | `"text" \| "image" \| "multimodal"` | `"text"` | Controls which document tools the agent can use |
| `system_prompt` | `str` | *(built-in)* | Custom system prompt for the agent |
| `max_iterations` | `int` | `50` | Maximum agent steps before raising `ExtractionError` |
| `max_retries` | `int` | `3` | Retry attempts on transient model failures |

---

## Modalities

Both extractors accept a `modality` parameter that controls how document content is presented to the model:

| Modality | What the model sees | When to use |
|---|---|---|
| `"text"` | Extracted text only | Reliable text layer, cost-sensitive, fast |
| `"image"` | Rendered page images (base64 PNG) | Visually rich documents, layout matters |
| `"multimodal"` | Text followed by images | Maximum signal, when accuracy justifies cost |

For the agentic extractor, modality controls which *tools* are exposed: `"text"` provides `read_text`, `"image"` provides `load_images`, and `"multimodal"` provides both. `get_page_count` is always available.

---

## Error handling

All document-level failures derive from `DocumentLoadError`, making them easy to catch together or individually. Extraction failures raise `ExtractionError`.

```python
from agentic_kie import (
    DocumentLoadError,
    CorruptDocumentError,
    PasswordProtectedError,
    EmptyDocumentError,
    OCRNotConfiguredError,
    ExtractionError,
)

try:
    doc = loader.load(path)
    result = extractor.extract(doc)
except PasswordProtectedError:
    ...  # encrypted PDF
except OCRNotConfiguredError:
    ...  # scanned document, no OCR provider
except EmptyDocumentError:
    ...  # zero pages or no extractable text
except CorruptDocumentError:
    ...  # unparseable file
except DocumentLoadError:
    ...  # catch-all for loading failures
except ExtractionError:
    ...  # agent exceeded iteration limit
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, available `make` targets, and the CI/CD pipeline.
