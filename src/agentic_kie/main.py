from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from nda import NDA

from agentic_kie.extractors.single_pass import SinglePassExtractor
from agentic_kie.loader import PDFLoader

load_dotenv()

file_path = (
    Path().cwd()
    / "notebooks"
    / "data"
    / "train"
    / "documents"
    / "0a68451dc19053b04342ce829bcd1321.pdf"
)


def main() -> None:
    model = ChatAnthropic(model="claude-haiku-4-5")

    loader = PDFLoader()
    extractor = SinglePassExtractor(model=model, schema=NDA)

    doc = loader.load(file_path)
    result = extractor.extract(doc)

    print(result.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
