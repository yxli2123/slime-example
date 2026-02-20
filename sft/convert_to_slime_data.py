import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


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
    parser.add_argument("--hf-data", type=str, required=True)
    parser.add_argument("--local-data", type=str, required=True)
    parser.add_argument("--input-key", type=str, default="messages")

    args = parser.parse_args()

    data = load_dataset(args.hf_data, split="train")
    print(f"Loaded {len(data)} examples from {args.hf_data}")

    filtered_data = []
    for d in data:
        keep_flag = True
        for msg in d[args.input_key]:
            if "role" not in msg:
                keep_flag = False
            if "content" not in msg or msg["content"] is None:
                keep_flag = False

        if keep_flag:
            filtered_data.append(d)
    save_dicts_to_jsonl(filtered_data, args.local_data)
    print(f"Saved {len(filtered_data)} examples to {args.local_data}")


if __name__ == "__main__":
    main()

