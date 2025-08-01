"""Preprocess generated records into instruction format."""

from __future__ import annotations

import json
from colorama import Fore
import yaml

with open("config.yaml", "r", encoding="utf-8") as cfg_file:
    config: dict = yaml.safe_load(cfg_file)

def main() -> None:
    """Create a flat instruction dataset from generated data."""
    instructions: list[dict[str, str]] = []
    with open(config["data"]["dataset_file"], "r") as f:
        data = json.load(f)
        for _, chunk in data.items():
            for pairs in chunk["generated"]:
                context_pair = {
                    "question": f"{pairs['question']}",
                    "answer": pairs["answer"],
                }
                instructions.append(context_pair)
            print(Fore.YELLOW + str(chunk))
            print("\n~~~~~~~~~~~~~~~~~~~~~")

    with open(config["data"]["instruction_file"], "w") as f:
        json.dump(instructions, f)

    with open(config["data"]["instruction_file"], "r") as f:
        data = json.load(f)
        print(Fore.LIGHTMAGENTA_EX + str(data[:10]))


if __name__ == "__main__":
    main()
