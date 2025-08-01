# Parameter Efficient Fine Tuning with LoRA

This project demonstrates how to fine tune a causal language model using [LoRA](https://arxiv.org/abs/2106.09685) and the HuggingFace ecosystem. It follows the approach shown in [this video](https://youtu.be/D3pXSkGceY0) but adapts it for this repository.

## Overview

1. **`syntheticdatageneration.py`** – Reads a PDF specified in `config.yaml`, splits it into chunks and uses an LLM to generate question/answer pairs. The output is saved as JSON.
2. **`preprocessing.py`** – Converts the generated JSON into a flat instruction dataset ready for training.
3. **`dataquality.py`** – Optional step that ranks instructions using another LLM and filters low quality entries.
4. **`train.py`** – Loads the instruction dataset and fine tunes the base model using LoRA. Hyperparameters such as rank, alpha and number of epochs are stored in `config.yaml`.

Each script can be run locally or on [RunPod](https://www.runpod.io/) with the same configuration files.

## Quickstart

1. Install dependencies and create a virtual environment using [`uv`](https://github.com/astral-sh/uv):
   ```bash
   uv init
   uv sync
   ```
2. Copy `.env.example` to `.env` and add your HuggingFace token.
3. Generate data:
   ```bash
   uv run syntheticdatageneration.py
   uv run preprocessing.py
   uv run dataquality.py  # optional
   ```
4. Train the model:
   ```bash
   uv run train.py
   ```

### Running on RunPod

Upload the repository and `pyproject_use_this_one_on_runpod.toml` (renamed to `pyproject.toml`), then run the same commands as above inside the pod.

## Configuration

Hyperparameters and file paths are defined in `config.yaml`:

```yaml
model:
  base_model: "meta-llama/Llama-3.2-1B"
  quant:
    load_in_4bit: true
    double_quant: true
    quant_type: nf4
    compute_dtype: bfloat16
lora:
  r: 256
  alpha: 512
  dropout: 0.05
trainer:
  epochs: 50
  output_dir: "meta-llama/Llama-3.2-1B-SFT"
data:
  pdf_path: "tm1_dg_dvlpr-10pages.pdf"
  dataset_file: "tm1data.json"
  instruction_file: "data/instruction.json"
```

Set your HuggingFace token in `.env`:

```bash
HF_TOKEN=your_hf_access_token
```

## Expected Output

After running the scripts you will obtain `data/instruction.json` containing question/answer pairs suitable for fine tuning. The trained model is saved under the directory specified by `trainer.output_dir`.

## License

This project is released under the MIT license.
