"""Training script for LoRA fine-tuning."""

from __future__ import annotations

import os
import yaml
from datasets import load_dataset
from colorama import Fore
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

load_dotenv()

with open("config.yaml", "r", encoding="utf-8") as cfg_file:
    config: dict = yaml.safe_load(cfg_file)

dataset = load_dataset("data", split="train")
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)

def format_chat_template(batch: dict, tokenizer: AutoTokenizer) -> dict:
    """Apply chat template to each question/answer pair.

    Parameters
    ----------
    batch : dict
        Batch of question/answer pairs from the dataset.
    tokenizer : AutoTokenizer
        Tokenizer used to apply the chat template.

    Returns
    -------
    dict
        Mapping containing instructions, responses and processed text.
    """

    system_prompt =  """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    samples = []

    # Access the inputs from the batch
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        # Apply chat template and append the result to the list
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples  # The processed chat template text for each row
    }

base_model = config["model"]["base_model"]
tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
)

train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=8, batched=True, batch_size=10)
print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET) 


quant_cfg = config["model"]["quant"]
quant_config = BitsAndBytesConfig(
    load_in_4bit=quant_cfg["load_in_4bit"],
    bnb_4bit_use_double_quant=quant_cfg["double_quant"],
    bnb_4bit_quant_type=quant_cfg["quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, quant_cfg["compute_dtype"]),
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="cuda:0",
    quantization_config=quant_config,
    token=os.getenv("HF_TOKEN"),
    cache_dir="./workspace",
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir=config["trainer"]["output_dir"],
        num_train_epochs=config["trainer"]["epochs"],
    ),
    peft_config=peft_config,
)

def main() -> None:
    """Run the training pipeline."""
    trainer.train()
    trainer.save_model("complete_checkpoint")
    trainer.model.save_pretrained("final_model")


if __name__ == "__main__":
    main()