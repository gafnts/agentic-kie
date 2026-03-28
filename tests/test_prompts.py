from agentic_kie.prompts import SINGLE_PASS_SYSTEM_PROMPT


class TestSinglePassSystemPrompt:
    def test_is_nonempty_string(self) -> None:
        assert isinstance(SINGLE_PASS_SYSTEM_PROMPT, str)
        assert len(SINGLE_PASS_SYSTEM_PROMPT) > 0

    def test_mentions_extraction(self) -> None:
        assert "extract" in SINGLE_PASS_SYSTEM_PROMPT.lower()

    def test_instructs_null_for_missing_fields(self) -> None:
        assert "null" in SINGLE_PASS_SYSTEM_PROMPT.lower()
