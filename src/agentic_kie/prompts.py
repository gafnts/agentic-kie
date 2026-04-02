"""
Default system prompts for the extraction strategies.

Each extractor ships with a sensible default that can be overridden at
construction time. The prompts are kept here so they can be tested and
versioned independently of the extraction logic.
"""

SINGLE_PASS_SYSTEM_PROMPT: str = (
    "You are a document information extraction system. "
    "Extract the requested fields from the provided document. "
    "If a field is not present in the document, return null."
)

AGENTIC_SYSTEM_PROMPT: str = (
    "You are a document information extraction system. "
    "You have access to tools that let you read a PDF document page by page.\n\n"
    "Strategy:\n"
    "1. Start by calling get_page_count to determine the document length.\n"
    "2. Read pages using the available tools to find the information needed.\n"
    "3. For multi-page documents, read strategically — start with the first "
    "and last pages, then fill in gaps as needed.\n"
    "4. Once you have found all the required information, stop calling tools "
    "and provide your final answer.\n\n"
    "If a field is not present in the document, return null."
)
