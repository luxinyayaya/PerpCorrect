import os
import wandb
import torch
import random
from fire import Fire
from trl import DPOTrainer, RewardTrainer, RewardConfig
from copy import deepcopy
from module import calc_ppl, GMM
from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="SEQ_CLS",
)


def data_filp(x):
    x["chosen"], x["rejected"] = x["rejected"], x["chosen"]
    if ("chosen_ppl" in x) and ("rejected_ppl" in x):
        x["chosen_ppl"], x["rejected_ppl"] = x["rejected_ppl"], x["chosen_ppl"]
    x["filped"] = 1 - x["filped"]
    return x


def prapare_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def prepare_dataset(data_path, gemma, epsilon, seed):
    random.seed(seed)
    dataset = load_dataset(data_path)
    dataset["train"] = dataset["train"].shuffle(seed=seed)
    warmup_dataset = dataset["train"].select(range(gemma))
    dataset["train"] = dataset["train"].add_column(
        "filped", [0 for _ in range(dataset["train"].num_rows)]
    )
    dataset["train"] = dataset["train"].map(
        lambda x: x if random.random() >= epsilon else data_filp(x)
    )
    return dataset["train"], dataset["test"], warmup_dataset


def ourdefense(
    model_path,
    train_dataset,
    test_dataset,
    warmup_dataset,
    save_dir,
):

    data_train, data_warmup = None, deepcopy(warmup_dataset)
    sum_size, delta_size = 0, round(0.02 * train_dataset.num_rows)
    best_data_train, best_threshold, best_detach, best_alpha = None, 0, 0, 0

    for i in range(6):
        model, tokenizer = prapare_model(model_path)
        save_path = os.path.join(save_dir, f"iter_epoch{i}")
        training_args = TrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            logging_steps=10,
            save_strategy="no",
            evaluation_strategy="epoch",
            output_dir=save_path,
            report_to="wandb",
            run_name=f"iter {save_path}",
        )
        dpo_trainer = DPOTrainer(
            model,
            args=training_args,
            beta=0.1,
            peft_config=peft_config,
            train_dataset=data_warmup,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            max_length=1024,
            max_prompt_length=512,
            max_target_length=512,
        )
        dpo_trainer.train()

        def map_ppl(sample):
            return {
                "prompt": sample["prompt"],
                "filped": sample["filped"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "chosen_ppl": calc_ppl(
                    model, tokenizer, sample["prompt"], sample["chosen"]
                ),
                "rejected_ppl": calc_ppl(
                    model, tokenizer, sample["prompt"], sample["rejected"]
                ),
            }

        data_train = deepcopy(train_dataset)
        data_train = data_train.map(map_ppl)
        spliter = GMM(
            data_train,
            "chosen_ppl",
            "rejected_ppl",
            "filped",
        )
        try:
            spliter.push_to_hub(save_path.replace("/", "-"))
        except Exception as e:
            print(e)
        threshold, detach = spliter.get_detach()
        alpha_fit = spliter.alpha_fit

        if detach > best_detach:
            best_data_train, best_threshold, best_detach, best_alpha = (
                deepcopy(data_train),
                threshold,
                detach,
                alpha_fit,
            )

        sum_size = min(sum_size + delta_size, train_dataset.num_rows)
        left_bound, right_bound = spliter.calc_bound(sum_size)
        alpha = data_train.filter(
            lambda x: x["chosen_ppl"] - x["rejected_ppl"] < left_bound
        ).select_columns(["prompt", "chosen", "rejected"])
        beta = (
            data_train.filter(
                lambda x: x["chosen_ppl"] - x["rejected_ppl"] > right_bound
            )
            .map(data_filp)
            .select_columns(["prompt", "chosen", "rejected"])
        )
        data_warmup = deepcopy(warmup_dataset)
        data_warmup = concatenate_datasets([data_warmup, alpha, beta])
        wandb.finish()

    try:
        dataset = {"train": best_data_train, "test": test_dataset}
        dataset = DatasetDict(dataset)
        dataset.push_to_hub(save_dir.replace("/", "-"), private=True)
    except Exception as e:
        print(e)

    best_data_train = best_data_train.map(
        lambda x: (
            x
            if (x["chosen_ppl"] - x["rejected_ppl"] < best_threshold)
            else data_filp(x)
        )
    )

    return best_data_train, 1 - best_alpha


def main(
    model_path: str,
    data_path: str,
    save_dir: str,
    lora_path: str,
    epsilon: float,
    defense: str,
    seed: int,
):

    save_dir = os.path.join(
        save_dir,
        model_path.split("/")[-1],
        data_path.split("/")[-1],
        f"{defense}-{epsilon}-{seed}",
    )
    try:
        os.makedirs(save_dir)
    except Exception as e:
        print(e)

    train_dataset, test_dataset, warmup_dataset = prepare_dataset(
        data_path, 50, epsilon, seed
    )

    # defense
    if "ours" in defense:
        train_dataset, epsilon = ourdefense(
            model_path,
            train_dataset,
            test_dataset,
            warmup_dataset,
            save_dir,
        )
    if "cdpo" in defense:
        label_smoothing = epsilon
    elif "rdpo" in defense:
        label_smoothing = -epsilon
    else:
        label_smoothing = 0

    # train
    training_args = RewardConfig(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        remove_unused_columns=False,
        gradient_accumulation_steps=5,
        max_length=1024,
        label_smoothing_factor=label_smoothing,
        learning_rate=3e-4,
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="epoch",
        output_dir=save_dir,
        report_to="wandb",
        run_name=f"ppo {save_dir}",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        is_trainable=False,
    )
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(
            examples["prompt"], examples["chosen"], examples["rejected"]
        ):
            tokenized_j = tokenizer(prompt + chosen, truncation=True, max_length=1024)
            tokenized_k = tokenizer(prompt + rejected, truncation=True, max_length=1024)
            new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_k["attention_mask"]
            )

        return new_examples

    train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    test_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=4)

    reward_trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
    )

    reward_trainer.train()
    reward_trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    Fire(main)
