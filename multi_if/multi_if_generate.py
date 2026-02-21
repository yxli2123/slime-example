from typing import List, TypedDict, Literal

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


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


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt.
    instruction: Instruction = sample.metadata
    init_prompt: List[Message] = [instruction["prompt"][0]]
    follow_up_prompt: List[Message] = instruction["prompt"][1:]
    total_turns = len(init_prompt) + len(follow_up_prompt)

    # Apply chat template for the init prompt.
    init_prompt_str_templated: str = state.tokenizer.apply_chat_template(init_prompt, tokenize=False, add_generation_prompt=True)
    init_prompt_ids_templated: List[int] = state.tokenizer(init_prompt_str_templated, add_special_tokens=False)["input_ids"]

    response_str_templated: str = ""
    response_ids_templated: List[int] = []
    loss_masks: List[Literal[0, 1]] = []

    for turn_id in range(total_turns):

        payload = {
            "text": init_prompt_str_templated + response_str_templated,
            "sampling_params": sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response_str_templated = output["text"]

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_ids_templated = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
        else:
            cur_response_ids_templated = state.tokenizer(cur_response_str_templated, add_special_tokens=False)["input_ids"]

        response_str_templated += cur_response_str_templated
        response_ids_templated += cur_response_ids_templated
        loss_masks += [1] * len(cur_response_ids_templated)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        if turn_id  < total_turns - 1:
            follow_up_prompt: List[Message] = [follow_up_prompt[turn_id]]
            follow_up_prompt_str_templated = state.tokenizer.apply_chat_template(follow_up_prompt, tokenize=False, add_generation_prompt=True)
            follow_up_prompt_ids_templated = state.tokenizer(follow_up_prompt_str_templated, add_special_tokens=False)["input_ids"]

            response_str_templated += follow_up_prompt_str_templated
            response_ids_templated += follow_up_prompt_ids_templated
            loss_masks += [0] * len(follow_up_prompt_ids_templated)

            # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
            # Check if maximum tool call count reached
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(follow_up_prompt_ids_templated)

                assert len(response_ids_templated) == len(
                    sample.rollout_log_probs
                ), f"Token/logp length mismatch at turn {turn_id}: {len(response_ids_templated)} tokens vs {len(sample.rollout_log_probs)} logps"

    # Set sample attributes
    sample.prompt = init_prompt_str_templated
    sample.tokens = init_prompt_ids_templated + response_ids_templated
    sample.response_length = len(response_ids_templated)
    sample.response = response_str_templated
    sample.loss_mask = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = init_prompt_str_templated + response_str_templated
    sample.payload_has_system = "<|im_start|>system" in sample.payload_text
    sample.payload_has_tools = "# Tools" in sample.payload_text

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample
