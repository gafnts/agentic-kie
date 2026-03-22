from pathlib import Path

from agentic_kie.reader import PDFReader

file_path = (
    Path().cwd()
    / "notebooks"
    / "data"
    / "train"
    / "documents"
    / "00a1d238e37ac225b8045a97953e845d.pdf"
)


def main() -> None:
    reader = PDFReader(file_path)
    print(reader.get_all_images())


if __name__ == "__main__":
    main()
