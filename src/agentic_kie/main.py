from pathlib import Path

from agentic_kie.loader import PDFLoader

file_path = (
    Path().cwd()
    / "notebooks"
    / "data"
    / "train"
    / "documents"
    / "724a6c9ea428637bd128cafb401b8a7e.pdf"
)


def main() -> None:
    pdf_document = PDFLoader().load(file_path)
    print(pdf_document.full_text)


if __name__ == "__main__":
    main()
