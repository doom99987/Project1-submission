"""
LoRA Fine-Tuning Script — finetune.py

Fine-tunes a small LLM (Mistral-7B-Instruct or LLaMA-3.1-8B-Instruct) on
synthetic game recommendation Q&A pairs using QLoRA (4-bit + LoRA adapters).

Run this on the GPU server (inference.ai):
    pip install -r requirements_finetune.txt
    python finetune.py

Output: ./game-rec-adapter/  (LoRA adapter weights, ~50MB)
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # ~14GB in 4-bit
# Alternative: "meta-llama/Meta-Llama-3.1-8B-Instruct" (requires HF token)
OUTPUT_DIR = "./game-rec-adapter"
TRAINING_DATA = "training_data.jsonl"
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 2
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_training_data(path: str) -> Dataset:
    """Load JSONL training data and format as chat messages."""
    records = []
    with open(path) as f:
        for line in f:
            record = json.loads(line.strip())
            # Use the messages format
            records.append({"messages": record["messages"]})
    print(f"Loaded {len(records)} training examples.")
    return Dataset.from_list(records)


def main():
    print("=" * 60)
    print("Game Recommendation Agent — LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Device:      {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow!)'}")
    print()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit quantization config (QLoRA) ────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load base model in 4-bit ──────────────────────────────────────────────
    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── LoRA config ───────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading training data...")
    dataset = load_training_data(TRAINING_DATA)

    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )}

    dataset = dataset.map(format_chat)
    print(f"Training on {len(dataset)} examples.")
    print("Sample:\n", dataset[0]["text"][:300])

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nDone! Adapter saved to: {OUTPUT_DIR}")
    print("Adapter size:", sum(
        f.stat().st_size for f in Path(OUTPUT_DIR).rglob("*") if f.is_file()
    ) // (1024 * 1024), "MB")


if __name__ == "__main__":
    main()
