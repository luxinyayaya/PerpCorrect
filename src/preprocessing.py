import re
import datasets
from fire import Fire


def load_and_unite(load_name):
    dataset = datasets.load_dataset(load_name)
    filter, converter = None, None
    if load_name == "Unified-Language-Model-Alignment/Anthropic_HH_Golden":

        def hhgolden_filter(sample):
            if sample["chosen"].count("Human:") != 1:
                return False
            if sample["chosen"].count("Assistant:") != 1:
                return False
            if sample["rejected"].count("Human:") != 1:
                return False
            if sample["rejected"].count("Assistant:") != 1:
                return False
            return True

        def extract_anthropic_prompt(prompt_and_response):
            """Extract the anthropic prompt from a prompt and response pair."""
            search_term = "\n\nAssistant: "
            search_term_idx = prompt_and_response.rfind(search_term)
            assert (
                search_term_idx != -1
            ), f"Prompt and response does not contain '{search_term}'"
            return prompt_and_response[: search_term_idx + len(search_term)]

        def hhgolden_converter(sample):
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "prompt": prompt.removeprefix("\n\nHuman: ").removesuffix(
                    "\n\nAssistant: "
                ),
                "chosen": sample["chosen"][len(prompt) :],
                "rejected": sample["rejected"][len(prompt) :],
            }

        filter, converter = hhgolden_filter, hhgolden_converter

    if load_name == "tasksource/oasst1_pairwise_rlhf_reward":

        def oasst1_filter(sample):
            return True

        def oasst1_converter(sample):
            return {
                "prompt": sample["prompt"].removeprefix("prompter: "),
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
            }

        filter, converter = oasst1_filter, oasst1_converter

    dataset = dataset.filter(filter)
    dataset = dataset.map(converter)

    def length_filter(sample):
        if len(sample["prompt"]) <= 1:
            return False
        if len(sample["rejected"]) <= 1:
            return False
        if len(sample["chosen"]) <= 1:
            return False
        return True

    dataset = dataset.filter(length_filter)
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])

    return dataset


def main(load_name, save_name):
    dataset = load_and_unite(load_name)

    train = dataset["train"]
    test = dataset[
        (
            "validation"
            if load_name == "tasksource/oasst1_pairwise_rlhf_reward"
            else "test"
        )
    ]
    dataset = datasets.DatasetDict({"train": train, "test": test})
    dataset.push_to_hub(save_name, private=True)


if __name__ == "__main__":
    main("Unified-Language-Model-Alignment/Anthropic_HH_Golden", "hhgolden")
    main("tasksource/oasst1_pairwise_rlhf_reward", "oasst1")
