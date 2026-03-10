import functools
import json
import logging
import asyncio
import builtins
import inspect
import random
import re
import subprocess
import sys
from typing import Any, List, TypedDict, Literal
import signal
from datasets import load_dataset

from openai import OpenAI
from aws_bedrock_token_generator import provide_token

from slime.utils.types import Sample

PROMPT_SOFT_EVALUATION = """Your task is to verify if the given response satisfies the given constraint. 
You should also verify if the given response makes sense and is generally helpful. 

If the response does not satisfy the constraint, return no.
If the response satisfies the constraint, but is not helpful, return no.
Otherwise, return yes.
Wrap your verification (yes or no) in a XML tag:
<answer>yes or no</answer>

**Constraint**
{constraint}

**Response**
{response}
"""

PROMPT_HELPFULNESS_EVALUATION = """Your task is to verify if the given response makes sense and is generally helpful. 
Unhelpful responses include but are not limited to severely incomplete sentences, keeping repeating the same content, or overly simplified responses.
Unless the constraints require the response to do so, above is considered as unhelpful.
You do not need to verify if the response satisfies the given constraints.

If the given response is unhelpful, return no. Otherwise, return yes.
Wrap your verification (yes or no) in a XML tag:
<answer>yes or no</answer>

**Constraint**
{constraint}

**Response**
{response}
"""


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class Constraint(TypedDict):
    id: str
    constraint: str
    func: List[str]

    verifier_type: str
    type: str
    description: str
    content_template: bool
    verifier_prompt: str

    quality: int
    frequency: int

class Verifier(TypedDict):
    id: str
    placeholder: str

class Persona(TypedDict):
    persona: str

class Instruction(TypedDict):
    id: str
    # assert len(prompt) == len(verifiers)
    prompt: List[Message]
    verifier: List[List[Verifier]]
    persona: str


def timeout(seconds: float):
    """Raise TimeoutError if the wrapped function runs longer than `seconds`.
    Unix-only (posix). Must be called from the main thread.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(signal, "SIGALRM"):
                raise RuntimeError("signal-based timeout requires Unix (SIGALRM).")

            def _handle_alarm(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds} seconds")

            old_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, _handle_alarm)
                # setitimer supports fractional seconds
                signal.setitimer(signal.ITIMER_REAL, seconds)
                return func(*args, **kwargs)
            finally:
                # always clean up timer & restore handler
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


def extract_content_within_tag(
    text: str,
    tag: str | None = None,
    tag_pair: tuple[str, str] | None = None,
    strict: bool = True,
) -> str | None:
    assert tag is not None or tag_pair is not None, "Use either tag or tag_pair"

    tag_l, tag_r = tag_pair if tag_pair else (f"<{tag}>", f"</{tag}>")
    pattern = rf"{re.escape(tag_l)}\s*(.*?)\s*{re.escape(tag_r)}"
    text_match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    default = "" if strict else text
    target_text = text_match.group(1) if text_match else default
    text = target_text if target_text else default

    return text


def build_id_to_data(data, id_name="id", muted=False):
    # Build ID -> Row dict from raw data.
    if not muted:
        print("Building id -> row map from prompt data ...")
    id2row = {}
    for raw_row in data:
        id2row[str(raw_row[id_name])] = raw_row
    if not muted:
        print(f"Mapped {len(id2row):,} ids.")

    return id2row


def execute_function(
    func_src: str,
    args: dict[str, Any],
    func_name: str = "evaluate",
) -> Any:
    """
    Compile `func_src`, locate the function `func_name` (or the first function defined),
    and call it with kwargs from `args`. Extra keys in `args` are ignored.
    Supports both def and async def.
    """

    # Give user code full access to Python builtins (HIGH RISK).
    # This includes `__import__`, so `import ...` inside func_src will work.
    g: dict[str, Any] = {"__builtins__": builtins}

    code = compile(func_src, filename="<user_function>", mode="exec")
    exec(code, g, g)

    if func_name not in g or not callable(g[func_name]):
        raise ValueError(f"Function '{func_name}' not found after compiling source.")

    fn = g[func_name]

    # Filter kwargs to the function signature
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in args.items() if k in sig.parameters}
    sig.bind(**filtered)  # raises if required args are missing

    # Call (await if coroutine)
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(**filtered))

    return fn(**filtered)


WORKER_PATH = "/root/slime/examples/multi_if/user_func_worker.py"
def execute_function_subprocess(
    func_src: str,
    args: dict[str, Any],
    func_name: str = "evaluate",
    *,
    timeout: float | None = 10.0,
    python_executable: str = sys.executable,
) -> bool:
    """
    Run user function in a separate Python process via `user_func_worker`.

    - func_src: source code containing the function definition.
    - args: kwargs passed to the function.
    - func_name: name of the function to call (default: "evaluate").
    - timeout: seconds before killing the subprocess (None = no timeout).
    - python_executable: which Python to use (default: current interpreter).

    Returns: the value returned by the user function (must be JSON-serializable
    or at least convertible via repr).
    """

    payload = {
        "func_src": func_src,
        "args": args,
        "func_name": func_name,
    }

    # Call the worker module as a script: `python -m user_func_worker`
    proc = subprocess.run(
        [python_executable, WORKER_PATH],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )

    # If the process died with a non-zero exit code, surface stderr
    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Worker process exited with code {proc.returncode}.\n"
            f"stderr:\n{stderr_text}"
        )

    stdout_text = proc.stdout.decode("utf-8", errors="replace").strip()
    if not stdout_text:
        raise RuntimeError("Worker produced no output.")

    resp = extract_content_within_tag(stdout_text, tag="answer")

    if resp.strip().lower() not in ["true", "false"]:
        logging.warning(f"Unable to extract answer. Got {resp}")

    return True if resp.strip().lower() == "true" else False


def call_api(
    base_url,
    api_key,
    model_name,
    prompt,
    temperature,
    max_tokens,
    top_p,
    n,
    idx: int = 0,
):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = []
    for _ in range(n):
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
        )
        resp = chat_response.choices[0].message.content
        response.append(resp)
    return idx, response


def evaluate(
    response: str,
    verifier: List[Verifier],
    constraint_pool: dict[str, dict],
    base_url,
    api_key,
    model_name: str = "openai.gpt-oss-20b-1:0",
    verify_helpfulness_rate: float = 0.0
) -> list[bool]:

    results = []
    for c in verifier:
        _id, _placeholder = c["id"], c["placeholder"]
        constraint_info = constraint_pool[_id]
        verifier_type = constraint_info["verifier_type"]
        content_template: bool = constraint_info["content_template"]
        func_input = {"response": response, "placeholder": _placeholder} if content_template else {"response": response}
        inner_result = []
        constraint = constraint_info["constraint"]

        if verifier_type == "hard":
            func: list[str] = constraint_info["func"]
            for f in func:
                try:
                    # answer = execute_function(f, func_input)
                    answer = execute_function_subprocess(f, func_input)
                except Exception as e:
                    logging.error(f"Failed to evaluate the function: {e}")
                    answer = False
                inner_result.append(answer)

            answer = all(inner_result)

            do_helpfulness = random.random() < verify_helpfulness_rate
            if do_helpfulness:
                prompt = [
                    {
                        "role": "user",
                        "content": PROMPT_HELPFULNESS_EVALUATION.format(response=response, constraint=constraint),
                    }
                ]
                try:
                    _, answer_list = call_api(base_url, api_key, model_name, prompt, 0.1, 2048, 0.95, 1)
                    answer_str: str = answer_list[0]
                    answer_str = extract_content_within_tag(answer_str, "answer")

                    logging.info(f"===>Helpfulness API result: {answer_str}")

                    is_helpful = True if answer_str is not None and answer_str.lower() == "yes" else False
                except Exception as e:
                    logging.error(f"Failed to call judge: {e}")
                    is_helpful = True

                answer = answer if is_helpful else False

            results.append(answer)

        else:
            prompt = [
                {
                    "role": "user",
                    "content": PROMPT_SOFT_EVALUATION.format(response=response, constraint=constraint),
                }
            ]
            try:
                _, answer_list = call_api(base_url, api_key, model_name, prompt, 0.1, 2048, 0.95, 1)
                answer_str: str = answer_list[0]
                answer_str = extract_content_within_tag(answer_str, "answer")

                logging.info(f"===> Soft API result: {answer_str}")

                answer = True if answer_str is not None and answer_str.lower() == "yes" else False
            except Exception as e:
                logging.error(f"Failed to call judge: {e}")
                answer = True

            results.append(answer)

    return results


CONSTRAIN_POOL = load_dataset("yxli2123/verifiable-constraints-1126", split="train")
INDEXED_CONSTRAIN_POOL = build_id_to_data(CONSTRAIN_POOL, muted=True)


async def reward_func(args, sample, **kwargs) -> float:
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Extract the response in multi-turns.
    completion_str_templated: str = sample.prompt + sample.response
    pattern = r"<\|im_start\|\>assistant\n\s*(.*?)\s*<\|im_end\|\>"
    response: List[str] = re.findall(pattern, completion_str_templated, flags=re.DOTALL)

    # Set up verifiers and passed.
    verifier: List[List[Verifier]] = sample.metadata["verifier"]
    if len(verifier) != len(response):
        logging.error(f"{len(response)} response != {len(verifier)} turns.")
        return 0.0

    passed = []

    if args.judge_api_key_path is not None:
        with open(args.judge_api_key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
    elif isinstance(args.judge_api_key, str) is not None:
        api_key = args.judge_api_key
    else:
        raise ValueError("Must provide the API key or a txt that stores it.")

    # Start to evaluate.
    for resp, ver in zip(response, verifier):
        turn_passed = evaluate(
            response=resp,
            verifier=ver,
            constraint_pool=INDEXED_CONSTRAIN_POOL,
            base_url=args.judge_base_url,
            api_key=api_key,
            verify_helpfulness_rate=args.verify_helpfulness_rate,
        )
        passed.append(all(turn_passed))

    return 1.0 if sum(passed) == len(passed) else 0.0


async def dumpy_reward_func(args, sample, **kwargs) -> float:
    return random.choice([0.0, 1.0])