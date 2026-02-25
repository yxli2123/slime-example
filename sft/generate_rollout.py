import logging

from slime.utils.processing_utils import load_processor, load_tokenizer
from transformers import AutoTokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        self.tokenizer = tokenizer
        self.system_message_length, self.gen_token_length = self.get_system_message_length()
        self.tokenizer_type = tokenizer_type

    @staticmethod
    def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
        return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]

    @staticmethod
    def find_all_sublist_indices(main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, tools=tools)["input_ids"]
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True)["input_ids"]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen_simple(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        """This function implements the loss mask of multi-turn loss that has no thinking, no tool use,
        and fixed system prompt (no system prompt template).
        """
        all_loss_masks = []
        all_token_ids = []

        if tools is not None:
            logging.warning("Find tools, but this implementation does not support tools.")

        for i, msg in enumerate(messages):
            # Ignore the system prompt is it is not the first in the `messages`.
            if msg["role"] == "system" and i == 0:
                message_ids = self.tokenizer.apply_chat_template([msg], tokenize=True)["input_ids"]
                loss_mask = [0] * len(message_ids)
            elif msg["role"] == "user":
                message_ids = self.tokenizer.apply_chat_template([msg], tokenize=True, add_generation_prompt=True)["input_ids"]
                loss_mask = [0] * len(message_ids)
            elif msg["role"] == "assistant":
                dump_user_query = "hello world"
                message_ids_combined = self.tokenizer.apply_chat_template([dump_user_query, msg], tokenize=True)["input_ids"]
                message_ids_users = self.tokenizer.apply_chat_template([dump_user_query], tokenize=True, add_generation_prompt=True)["input_ids"]
                message_ids = message_ids_combined[len(message_ids_users):]
                loss_mask = [1] * len(message_ids)
            else:
                raise NotImplementedError(f"The role `{msg['role']}` is not supported. Please filter your dataset before training.")

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True)

        for i, message in enumerate(messages):
            if i == 0:
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message], tokenize=True, tools=tools
                )["input_ids"]
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                prefixed_message_ids = self.tokenizer.apply_chat_template([prefix_message, message], tokenize=True)["input_ids"]
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        elif self.tokenizer_type == "qwen_simple":
            return self.gen_multi_turn_loss_mask_qwen_simple(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        text = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text.append({"role": msg["role"], "content": " ".join(text_parts)})
            else:
                text.append(msg)

        _, loss_mask_text = self.get_loss_mask(text, tools=tools)

        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        loss_mask = [0] * diff + loss_mask_text

        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts



TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[Sample]: a list of samples generated by the rollout
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        # TODO: THIS IS HARD.
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type="qwen_simple")

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        tools = sample.metadata.get("tools", None)

        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)

        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(f"sft_rollout::generate_rollout example sample: {sample=}")
            logger.info(f"sft_rollout::generate_rollout example messages: {messages=}")
            logger.info(f"sft_rollout::generate_rollout example token_ids: {token_ids=}")
            logger.info(f"sft_rollout::generate_rollout example loss_mask: {loss_mask=}")
            logger.info(f"sft_rollout::generate_rollout example response_length: {response_length=}")
            SAMPLE_PRINTED = True

    return samples



if __name__ == "__main__":
    # test_loss_mask_qwen3_simple("Qwen/Qwen3-8B")

    import argparse

    parser = argparse.ArgumentParser(description="Quick test for qwen_simple loss mask generation.")
    parser.add_argument(
        "--hf-checkpoint",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Hugging Face checkpoint for tokenizer loading.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    # NOTE: "qwen_simple" maps to "qwen3_simple" in this file.
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen_simple")

    example_messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4."},
        {"role": "user", "content": "Give one reason why."},
        {"role": "assistant", "content": "Because adding two pairs gives four items in total."},
    ]

    token_ids, loss_mask = mask_generator.get_loss_mask(example_messages)
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    response_length = mask_generator.get_response_lengths([loss_mask])[0]

    print("Token lengths", len(token_ids))
    print("Token IDs:")
    print(token_ids)
    print("\nLoss mask:")
    print(loss_mask)
    print("\nPer-token view (token_id, token, mask):")
    for idx, (token_id, token_text, mask) in enumerate(zip(token_ids, decoded_tokens, loss_mask)):
        print(f"{idx:04d}: {token_id:>8} | {token_text!r} | {mask}")
    print(f"\nResponse length: {response_length}")
