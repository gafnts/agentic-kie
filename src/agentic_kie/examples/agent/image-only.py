import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from nda import NDA

from agentic_kie.extractors.agent import AgenticExtractor
from agentic_kie.loader import PDFLoader

logging.basicConfig(level=logging.INFO)

load_dotenv()

file_path = (
    Path().cwd()
    / "notebooks"
    / "data"
    / "train"
    / "documents"
    / "00a1d238e37ac225b8045a97953e845d.pdf"
)


def main() -> None:
    claude = ChatAnthropic(model="claude-haiku-4-5")
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    gpt = ChatOpenAI(model_name="gpt-5.4")

    claude_extractor = AgenticExtractor(model=claude, schema=NDA, modality="image")
    gemini_extractor = AgenticExtractor(model=gemini, schema=NDA, modality="image")
    gpt_extractor = AgenticExtractor(model=gpt, schema=NDA, modality="image")

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
