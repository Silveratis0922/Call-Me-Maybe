"""Command line entry point: ``uv run python -m src``."""

import argparse
import sys
import time
from typing import Any, List, Optional

from pydantic import ValidationError

from src.decoder import ConstrainedDecoder
from src.io_utils import load_functions, load_prompts, warn, write_results
from src.models import CliArgs, FunctionCall

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args(argv: Optional[List[str]] = None) -> CliArgs:
    """Parse and validate the command line.

    Args:
        argv: Arguments to parse (defaults to ``sys.argv[1:]``).

    Returns:
        The validated arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function "
                    "calls using constrained decoding.")
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS,
                        help=f"function definitions file "
                             f"(default: {DEFAULT_FUNCTIONS})")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"prompts file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"results file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", dest="model_name", default=DEFAULT_MODEL,
                        help=f"Hugging Face model id (default: "
                             f"{DEFAULT_MODEL})")
    parser.add_argument("--verbose", action="store_true",
                        help="show the token-by-token generation")
    namespace = parser.parse_args(argv)
    return CliArgs.model_validate(vars(namespace))


def load_model(model_name: str) -> Any:
    """Load the LLM through the provided SDK.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        A ``Small_LLM_Model`` instance.

    Raises:
        RuntimeError: If the SDK is missing or the model cannot be loaded.
    """
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as exc:
        raise RuntimeError(
            f"cannot import llm_sdk ({exc}); run 'uv sync' first") from None
    try:
        return Small_LLM_Model(model_name=model_name)
    except Exception as exc:
        raise RuntimeError(
            f"cannot load model '{model_name}': {exc}") from None


def run(args: CliArgs) -> int:
    """Process every prompt and write the results.

    Args:
        args: Validated command line arguments.

    Returns:
        The process exit code.
    """
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not prompts:
        warn("no prompt to process")
        try:
            write_results(args.output, [])
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    print(f"Loading model {args.model_name}...")
    try:
        model = load_model(args.model_name)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    decoder = ConstrainedDecoder(model=model, functions=functions,
                                 verbose=args.verbose)
    results: List[FunctionCall] = []
    start = time.perf_counter()
    for index, item in enumerate(prompts, start=1):
        try:
            call = decoder.call(item.prompt)
        except Exception as exc:
            warn(f"prompt {index} failed ({exc}): {item.prompt!r}")
            continue
        results.append(call)
        print(f"[{index}/{len(prompts)}] {item.prompt!r} -> "
              f"{call.name}({call.parameters})")
    elapsed = time.perf_counter() - start

    try:
        write_results(args.output, results)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(results)}/{len(prompts)} prompts processed in "
          f"{elapsed:.1f}s ({decoder.model_calls} model calls). "
          f"Results written to {args.output}")
    return 0 if len(results) == len(prompts) else 1


def main() -> int:
    """Program entry point.

    Returns:
        The process exit code.
    """
    try:
        return run(parse_args())
    except ValidationError as exc:
        print(f"Error: invalid arguments: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
