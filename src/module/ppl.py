import torch
import numpy as np


def build_tokenized_answer(tokenizer, prompt, answer):
    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
    full_input_ids = np.array(full_tokenized["input_ids"])
    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError(
            "Prompt input ids and answer input ids should have the same length."
        )
    response_token_ids_start_idx = len(prompt_input_ids)
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1
    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][
        :response_token_ids_start_idx
    ]
    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError(
            "Prompt input ids and attention mask should have the same length."
        )
    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][
        response_token_ids_start_idx:
    ]
    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )


def tokenize_row(tokenizer, prompt, response, max_length=512, max_prompt_length=256):
    all_tokens = build_tokenized_answer(tokenizer, prompt, response)
    # bos_token
    all_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + all_tokens[
        "prompt_input_ids"
    ]
    all_tokens["prompt_attention_mask"] = [1] + all_tokens["prompt_attention_mask"]
    # eos_token
    all_tokens["input_ids"].append(tokenizer.eos_token_id)
    all_tokens["attention_mask"].append(1)

    response_length = len(all_tokens["input_ids"])

    if len(all_tokens["prompt_input_ids"]) + response_length > max_length:
        for k in ["prompt_input_ids", "prompt_attention_mask"]:
            all_tokens[k] = all_tokens[k][-max_prompt_length:]

    if len(all_tokens["prompt_input_ids"]) + response_length > max_length:
        for k in ["input_ids", "attention_mask"]:
            all_tokens[k] = all_tokens[k][: max_length - max_prompt_length]

    all_sequence_tokens = {
        k: all_tokens[f"prompt_{k}"] + all_tokens[k]
        for k in ["input_ids", "attention_mask"]
    }

    return all_sequence_tokens


def calc_ppl(model, tokenizer, prompt, response, max_length=512, max_prompt_length=256):
    all_sequence_tokens = tokenize_row(
        tokenizer, prompt, response, max_length, max_prompt_length
    )
    input_ids = torch.tensor(all_sequence_tokens["input_ids"]).unsqueeze(0)
    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
    labels = input_ids[:, 1:].clone()
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    avg_all_logps = per_token_logps.sum(-1) / per_token_logps.shape[-1]
    return -avg_all_logps.item()
