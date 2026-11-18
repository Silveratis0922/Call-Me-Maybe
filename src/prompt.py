"""Construction of the text prompt given to the LLM."""

from typing import List

from src.models import FunctionDefinition

SYSTEM_MESSAGE = (
    "You translate the user request into a single function call.\n"
    "Pick the function that matches the request and extract its arguments "
    "from the request, copying text values exactly. When an argument is a "
    "pattern or a regex, write the regular expression matching what the "
    "user describes.\n"
    "Available functions:\n"
)


def describe_function(function: FunctionDefinition) -> str:
    """Render a compact one-line signature of a function.

    Args:
        function: The function to describe.

    Returns:
        A line such as ``- fn_add(a: number, b: number) -> number: Add.``
    """
    arguments = ", ".join(f"{name}: {spec.type}"
                          for name, spec in function.parameters.items())
    returns = f" -> {function.returns.type}" if function.returns else ""
    description = f": {function.description}" if function.description else ""
    return f"- {function.name}({arguments}){returns}{description}"


def build_prompt(functions: List[FunctionDefinition], request: str) -> str:
    """Build the full chat prompt in the Qwen chat format.

    The prompt ends at the beginning of the assistant answer, with thinking
    disabled, so that the next tokens are the JSON function call.

    Args:
        functions: The functions the model may choose from.
        request: The natural-language user request.

    Returns:
        The prompt text.
    """
    catalogue = "\n".join(describe_function(item) for item in functions)
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}{catalogue}\n"
        "Answer with JSON: "
        '{"name": <function name>, "parameters": {<arguments>}}'
        "<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
