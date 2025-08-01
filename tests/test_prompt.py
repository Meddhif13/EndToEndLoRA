import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from generated_prompt import prompt_template


def test_prompt_contains_data():
    text = prompt_template("sample", 2)
    assert "sample" in text
