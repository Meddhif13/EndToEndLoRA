"""Generate synthetic instruction/answer pairs from PDF content."""

from __future__ import annotations

import json
from typing import List

import yaml
from pydantic import BaseModel
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore
from litellm import completion
from generated_prompt import prompt_template

with open("config.yaml", "r", encoding="utf-8") as cfg_file:
    config: dict = yaml.safe_load(cfg_file)

class Record(BaseModel):
    """Single question/answer record."""

    question: str
    answer: str

class Response(BaseModel):
    """Model response schema for a batch of records."""

    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    """Request the LLM to generate question/answer pairs for a text chunk."""
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta 
    return json.loads(data)

def main() -> None:
    """Generate data from the configured PDF."""
    converter = DocumentConverter()
    doc = converter.convert(config["data"]["pdf_path"]).document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    dataset = {}
    for i, chunk in enumerate(chunks): 
            print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}…" + Fore.RESET)
            enriched_text = chunker.contextualize(chunk=chunk)
            print(Fore.LIGHTMAGENTA_EX + f"Contextualized Tex:\n{enriched_text[:300]}…" + Fore.RESET)

            data = llm_call(
                enriched_text
            )
            dataset[i] = {"generated":data["generated"], "context":enriched_text}
    
    with open(config["data"]["dataset_file"], "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    main()






