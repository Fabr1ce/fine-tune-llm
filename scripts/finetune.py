import argparse
import json
import torch
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
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        lines = content.split('\n')
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                prompt = obj["instruction"]
                output = obj["output"]
                rows.append({"text": f"{prompt}\n{output}"})
        except json.JSONDecodeError as e:
            print(f"Skipping line {i+1}: {e}")
            print(f"Content: {repr(line)}")
            continue
    
    print(f"Loaded {len(rows)} examples")
    return Dataset.from_list(rows)


def main(args):
    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=None  # Load on CPU first
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
    
    # Now move to GPU
    model = model.to("cuda")

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
        train_dataset=tokenized,
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
