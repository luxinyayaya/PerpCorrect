import os
import torch
from fire import Fire
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)


def main(model_path, data_path, save_dir):
    dataset = load_dataset(data_path, split="train")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    base_model = prepare_model_for_kbit_training(base_model)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    save_dir = os.path.join(
        save_dir, model_path.split("/")[-1], data_path.split("/")[-1]
    )
    try:
        os.makedirs(save_dir)
    except Exception as e:
        print(e)

    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=1500,
    )
    max_seq_length = 1024
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.model.save_pretrained(save_dir)


if __name__ == "__main__":
    Fire(main)
