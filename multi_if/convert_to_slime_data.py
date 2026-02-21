import argparse
import json
import os
import random
from pathlib import Path
from typing import TypedDict, List, Literal
from datasets import load_dataset


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class Verifier(TypedDict):
    id: str
    placeholder: str

class Instruction(TypedDict):
    id: str
    # assert len(prompt) == len(verifiers)
    prompt: List[Message]
    verifier: List[List[Verifier]]
    persona: str


def save_dicts_to_jsonl(data_list, filename):
    """
    Saves a list of dictionaries to a JSONL file.

    Args:
        data_list (list): A list of dictionaries to be saved.
        filename (str): The path to the output JSONL file.
    """
    os.makedirs(Path(filename).parent, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-train-data", type=str, required=True)
    parser.add_argument("--hf-eval-data", type=str, required=True)
    parser.add_argument("--local-train-data", type=str, required=True)
    parser.add_argument("--local-eval-data", type=str, required=True)
    args = parser.parse_args()

    train_data: List[Instruction] = load_dataset(args.hf_train_data, split="train")
    eval_data: List[Instruction] = load_dataset(args.hf_eval_data, split="train")

    def _make_slime_data(_hf_data):
        _slime_data = []
        for rec in _hf_data:
            record = {
                "prompt": rec["prompt"][0]["content"],
                "label": "0",
                "metadata": {
                    "id": rec["id"],
                    "prompt": rec["prompt"],
                    "verifier": rec["verifier"],
                    "persona": rec["persona"],
                }
            }
            _slime_data.append(record)

        return _slime_data

    slime_train_data = _make_slime_data(train_data)
    slime_eval_data = _make_slime_data(eval_data)

    random.shuffle(slime_train_data)
    random.shuffle(slime_eval_data)


    save_dicts_to_jsonl(slime_train_data, args.local_train_data)
    save_dicts_to_jsonl(slime_eval_data, args.local_eval_data)


if __name__ == "__main__":
    main()

