import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
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
    claude = ChatAnthropic(model="claude-haiku-4-5")
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    gpt = ChatOpenAI(model="gpt-5.4")

    claude_extractor = SinglePassExtractor(model=claude, schema=NDA)
    gemini_extractor = SinglePassExtractor(model=gemini, schema=NDA)
    gpt_extractor = SinglePassExtractor(model=gpt, schema=NDA)

    loader = PDFLoader()
    doc = loader.load(file_path)

    results = {
        "claude": claude_extractor.extract(doc),
        "gemini": gemini_extractor.extract(doc),
        "gpt": gpt_extractor.extract(doc),
    }

    print(
        json.dumps(
            {model: result.model_dump() for model, result in results.items()}, indent=4
        )
    )


if __name__ == "__main__":
    main()
