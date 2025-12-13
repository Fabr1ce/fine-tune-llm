# Fine-Tuning Llama-3-8B on AMD ROCm (LoRA)

# 📁 **Repo Structure**

```
llama3-rocm-finetune/
│
├── README.md
├── environment.yml
├── docker/
│   └── Dockerfile
├── scripts/
│   ├── finetune.py
│   ├── infer.py
│   ├── benchmark.py
│   └── prepare_data.py
├── data/
│   └── sample.jsonl
├── notebooks/
│   └── Finetune_Llama3_ROCm.ipynb
└── results/
    ├── logs/
    ├── checkpoints/
    └── samples/
```

---

## 🦙 Llama-3-8B ROCm Finetuning — LoRA on AMD GPUs

Fine-tune **Llama-3-8B** on **AMD ROCm** using LoRA.
Works on AMD MI250 / MI300 (AMD Developer Cloud).

This repo provides:

* 🚀 One-click training script
* 🎯 LoRA fine-tuning using 🤗 Transformers
* 🔥 Full ROCm support
* 📊 Benchmarking on AMD Cloud
* 🧪 Inference script + sample outputs
* 🧰 Dockerfile + conda environment

---

## 📦 Install environment

```bash
conda env create -f environment.yml
conda activate llama3-rocm
```

---

## ▶️ Prepare data

Place your instruction/prompt pairs in `data/sample.jsonl`.

Example:

```json
{"instruction": "Explain AMD GPUs to a 10 year old", "output": "AMD GPUs help computers think fast..."}
```

Run preprocessing (optional):

```bash
python scripts/prepare_data.py
```

---

## 🚀 Run finetuning

```bash
python scripts/finetune.py \
  --model meta-llama/Meta-Llama-3-8B \
  --data data/sample.jsonl \
  --output_dir results/checkpoints \
  --batch_size 2 \
  --lr 2e-4 \
  --epochs 1
```

Trains a small LoRA in under a few hours on MI250.

---

## 🧪 Inference (LoRA + base model)

```bash
python scripts/infer.py \
  --model meta-llama/Meta-Llama-3-8B \
  --lora results/checkpoints \
  --prompt "Explain ROCm to a beginner."
```

---

## 📊 Benchmarks

Add your AMD Cloud results:

| GPU    | Batch | Tokens/s | Time/Epoch | Cost |
| ------ | ----- | -------- | ---------- | ---- |
| MI250  | 2     | TBD      | TBD        | TBD  |
| MI300X | 4     | TBD      | TBD        | TBD  |

---

## 🐳 Optional: Docker (ROCm)

Build:

```bash
docker build -t llama3-rocm -f docker/Dockerfile .
```

Run:

```bash
docker run --device=/dev/kfd --device=/dev/dri -it llama3-rocm
```

---

# 🧠 **Training Script (finetune.py)**

```python
import argparse
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


def load_data(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj["instruction"]
            output = obj["output"]
            rows.append({"text": f"{prompt}\n{output}"})
    return Dataset.from_list(rows)


def main(args):
    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)

    dataset = load_data(args.data)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=True,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset(tokenized),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default="results/checkpoints")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args)
```

---

# 🧪 **Inference Script (infer.py)**

```python
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    model = PeftModel.from_pretrained(base, args.lora)

    prompt = args.prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    main(args)
```

---

# 🧱 **environment.yml**

```yaml
name: llama3-rocm
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - pytorch-cuda=none
  - pip
  - pip:
      - transformers
      - datasets
      - peft
      - accelerate
      - sentencepiece
```

---

# 🐳 **Dockerfile (ROCm)**

```dockerfile
FROM rocm/pytorch:latest

RUN pip install transformers datasets peft accelerate sentencepiece

WORKDIR /workspace
COPY . .
```

---
