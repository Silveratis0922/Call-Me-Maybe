"""Reading and writing of the JSON files with graceful error handling."""

import json
import os
import sys
from typing import Any, List

from pydantic import ValidationError

from src.models import FunctionCall, FunctionDefinition, PromptItem


def warn(message: str) -> None:
    """Print a warning on stderr.

    Args:
        message: Text to display.
    """
    print(f"[warning] {message}", file=sys.stderr)


def read_json(path: str) -> Any:
    """Load a JSON document from disk.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.

    Raises:
        ValueError: If the file is missing, unreadable or not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ValueError(f"file not found: '{path}'") from None
    except IsADirectoryError:
        raise ValueError(f"'{path}' is a directory, not a file") from None
    except PermissionError:
        raise ValueError(f"permission denied: '{path}'") from None
    except UnicodeDecodeError:
        raise ValueError(f"'{path}' is not valid UTF-8 text") from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in '{path}' (line {exc.lineno}, "
            f"column {exc.colno}): {exc.msg}"
        ) from None
    except OSError as exc:
        raise ValueError(f"cannot read '{path}': {exc.strerror}") from None


def load_functions(path: str) -> List[FunctionDefinition]:
    """Load and validate the function definitions file.

    Invalid entries are skipped with a warning.

    Args:
        path: Path to ``functions_definition.json``.

    Returns:
        The list of valid function definitions.

    Raises:
        ValueError: If the file cannot be used at all.
    """
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"'{path}' must contain a JSON array of functions")
    functions: List[FunctionDefinition] = []
    seen: set[str] = set()
    for index, raw in enumerate(data):
        try:
            function = FunctionDefinition.model_validate(raw)
        except ValidationError as exc:
            warn(f"skipping function #{index} in '{path}': "
                 f"{_format_validation_error(exc)}")
            continue
        if function.name in seen:
            warn(f"skipping duplicate function '{function.name}'")
            continue
        seen.add(function.name)
        functions.append(function)
    if not functions:
        raise ValueError(f"no valid function definition found in '{path}'")
    return functions


def load_prompts(path: str) -> List[PromptItem]:
    """Load and validate the prompts file.

    Invalid entries are skipped with a warning.

    Args:
        path: Path to ``function_calling_tests.json``.

    Returns:
        The list of valid prompts (possibly empty).

    Raises:
        ValueError: If the file cannot be used at all.
    """
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"'{path}' must contain a JSON array of prompts")
    prompts: List[PromptItem] = []
    for index, raw in enumerate(data):
        try:
            prompts.append(PromptItem.model_validate(raw))
        except ValidationError as exc:
            warn(f"skipping prompt #{index} in '{path}': "
                 f"{_format_validation_error(exc)}")
    return prompts


def write_results(path: str, results: List[FunctionCall]) -> None:
    """Write the results as a JSON array.

    Args:
        path: Destination file; parent directories are created.
        results: Function calls to serialise.

    Raises:
        ValueError: If the file cannot be written.
    """
    payload = [result.model_dump() for result in results]
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ValueError(f"cannot write '{path}': {exc.strerror}") from None


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a pydantic error into a short one-line message.

    Args:
        exc: The validation error.

    Returns:
        A readable summary of every problem found.
    """
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "entry"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
