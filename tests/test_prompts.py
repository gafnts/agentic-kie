from agentic_kie.prompts import AGENTIC_SYSTEM_PROMPT, SINGLE_PASS_SYSTEM_PROMPT


class TestSinglePassSystemPrompt:
    def test_is_nonempty_string(self) -> None:
        assert isinstance(SINGLE_PASS_SYSTEM_PROMPT, str)
        assert len(SINGLE_PASS_SYSTEM_PROMPT) > 0

    def test_mentions_extraction(self) -> None:
        assert "extract" in SINGLE_PASS_SYSTEM_PROMPT.lower()

    def test_instructs_null_for_missing_fields(self) -> None:
        assert "null" in SINGLE_PASS_SYSTEM_PROMPT.lower()


class TestAgenticSystemPrompt:
    def test_is_nonempty_string(self) -> None:
        assert isinstance(AGENTIC_SYSTEM_PROMPT, str)
        assert len(AGENTIC_SYSTEM_PROMPT) > 0

    def test_mentions_tools(self) -> None:
        assert "tools" in AGENTIC_SYSTEM_PROMPT.lower()

    def test_mentions_page_count(self) -> None:
        assert "get_page_count" in AGENTIC_SYSTEM_PROMPT

    def test_instructs_null_for_missing_fields(self) -> None:
        assert "null" in AGENTIC_SYSTEM_PROMPT.lower()
