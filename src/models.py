"""Pydantic data models describing the inputs and outputs of the program."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ParamType = Literal["number", "integer", "string", "boolean"]

TYPE_ALIASES: Dict[str, ParamType] = {
    "number": "number",
    "float": "number",
    "integer": "integer",
    "int": "integer",
    "string": "string",
    "str": "string",
    "boolean": "boolean",
    "bool": "boolean",
}


class ParameterSpec(BaseModel):
    """Type description of a single function argument or return value.

    Attributes:
        type: The JSON type of the value (aliases like ``int`` are
            normalised).
        description: Optional human readable description.
    """

    model_config = ConfigDict(extra="ignore")

    type: ParamType
    description: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def normalise_type(cls, value: Any) -> Any:
        """Map type aliases (``int``, ``str``...) to canonical names.

        Args:
            value: Raw ``type`` field read from the JSON file.

        Returns:
            The canonical type name, or the raw value if unknown (pydantic
            then reports a clear validation error).
        """
        if isinstance(value, str):
            return TYPE_ALIASES.get(value.strip().lower(), value)
        return value


class FunctionDefinition(BaseModel):
    """A function the LLM is allowed to call.

    Attributes:
        name: Unique function name.
        description: What the function does (shown to the LLM).
        parameters: Ordered mapping of argument name to its type.
        returns: Type of the returned value, if given.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ParameterSpec] = None


class PromptItem(BaseModel):
    """A single natural-language request to translate.

    Attributes:
        prompt: The user request.
    """

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):
    """The structured result produced for one prompt.

    Attributes:
        prompt: The original natural-language request.
        name: Name of the selected function.
        parameters: Arguments with values of the correct type.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, Any]


class CliArgs(BaseModel):
    """Validated command line arguments.

    Attributes:
        functions_definition: Path to the function definitions file.
        input: Path to the prompts file.
        output: Path of the results file to write.
        model_name: Hugging Face identifier of the model to load.
        verbose: Whether to print the generation process.
    """

    functions_definition: str
    input: str
    output: str
    model_name: str
    verbose: bool = False
