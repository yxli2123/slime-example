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
    parser.add_argument("--max-turns", type=int)

    args = parser.parse_args()

    data = load_dataset(args.hf_data, split="train")
    print(f"Loaded {len(data)} examples from {args.hf_data}")

    filtered_data = []
    for d in data:
        keep_flag = True
        word_cnt = 0
        for turn_id, msg in enumerate(d[args.input_key]):
            # If num of turns greater than args.max_turns, filter out.
            if args.max_turns and turn_id // 2 >= args.max_turns:
                keep_flag = False
                break

            if word_cnt >= 2024:
                keep_flag = False
                break

            # If not user and assistant in turn, filter out.
            if turn_id % 2 == 0 and msg.get("role") != "user":
                keep_flag = False
            elif turn_id % 2 == 1 and msg.get("role") != "assistant":
                keep_flag = False

            # If content is not a text, filter our.
            if not isinstance(msg.get("content"), str):
                keep_flag = False
            else:
                word_cnt += len(msg.get("content").split(" "))

        if keep_flag:
            filtered_data.append(d)
    save_dicts_to_jsonl(filtered_data, args.local_data)
    print(f"Saved {len(filtered_data)} examples to {args.local_data}")


if __name__ == "__main__":
    main()

