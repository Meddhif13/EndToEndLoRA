from litellm import completion
"""Example script for a direct LLM call."""

from __future__ import annotations
from colorama import Fore


def llm_call(prompt: str) -> None:
    """Stream the LLM response for the given prompt."""
    stream = completion(
        model="ollama_chat/tm1bud-dq300target:latest",
        # top_k=1,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )
    data = ""
    for x in stream:
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None:
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="")
            data += delta

def main() -> None:
    """Run a quick example call."""
    llm_call("In a TM1 cube, what's the minimum number of dimensions?")


if __name__ == "__main__":
    main()
