"""Token-level constraints used by the constrained decoder.

A constraint describes the set of byte strings a value may take (a JSON
number, a JSON string, one name among a list...). For every candidate token
it answers one question: "can this token extend what has been generated so
far while staying valid?". Tokens for which the answer is no are the ones
whose logit is set to minus infinity by the decoder.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, Field

NUMBER_COMPLETE = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
                             rb"(?:[eE][+-]?[0-9]+)?")
NUMBER_PREFIX = re.compile(rb"-?(?:(?:0|[1-9][0-9]*)"
                           rb"(?:\.[0-9]*|\.[0-9]+[eE][+-]?[0-9]*"
                           rb"|[eE][+-]?[0-9]*)?)?")
INTEGER_COMPLETE = re.compile(rb"-?(?:0|[1-9][0-9]*)")
INTEGER_PREFIX = re.compile(rb"-?(?:0|[1-9][0-9]*)?")

HEX_DIGITS = b"0123456789abcdefABCDEF"
SIMPLE_ESCAPES = b"\"\\/bfnrt"
QUOTE = ord('"')
BACKSLASH = ord("\\")

# String scanner states: NORMAL, after a backslash, or inside \uXXXX with
# HEX_BASE + n hexadecimal digits still expected.
NORMAL = 0
ESCAPE = 1
HEX_BASE = 10


class StepResult(BaseModel):
    """Outcome of feeding one token to a constraint.

    Attributes:
        prefix: Bytes of the token that belong to the value opener (an
            optional space, and the opening quote of a string). They are
            part of the generated text but not of the value.
        accepted: Bytes of the token kept in the value. It can be shorter
            than the token when the token also contains the end of the
            value (for instance ``3,`` for a number: only ``3`` is kept).
        done: True when the value is complete.
        started: True when the opener is complete after this step.
        state: Internal scanner state after the accepted bytes.
    """

    prefix: bytes = b""
    accepted: bytes = b""
    done: bool = False
    started: bool = False
    state: int = NORMAL


class Constraint(BaseModel, ABC):
    """Base class of every value constraint.

    A value is written as an optional space, then the ``opener`` (``"`` for
    strings), then the value itself. Letting the model generate the space
    and the opener keeps its natural tokenization: Qwen writes ``": -5"``
    with the single token ``' -'``, which would be impossible if the space
    was forced by the decoder.

    Attributes:
        opener: Bytes that must precede the value.
        prefix: Opener bytes generated so far.
        started: True once the opener is complete.
        buffer: Bytes generated so far for the value itself.
        state: Internal scanner state (only used by some constraints).
    """

    opener: bytes = b""
    prefix: bytes = b""
    started: bool = False
    buffer: bytes = b""
    state: int = NORMAL

    def _openers(self) -> List[bytes]:
        """Return the complete openers: with or without a leading space."""
        return [self.opener, b" " + self.opener]

    def check(self, piece: bytes) -> Optional[StepResult]:
        """Test whether a token may be appended to the generated text.

        Args:
            piece: Bytes of the candidate token.

        Returns:
            A StepResult if the token is allowed, None otherwise.
        """
        if not piece:
            return None
        if self.started:
            return self.check_value(piece)
        openers = self._openers()
        for index in range(len(piece) + 1):
            head = self.prefix + piece[:index]
            if not any(opener.startswith(head) for opener in openers):
                return None
            if head not in openers:
                continue
            rest = piece[index:]
            if not rest:
                return StepResult(prefix=piece, started=True)
            if head == b"" and rest.startswith(b" "):
                continue
            result = self.check_value(rest)
            if result is None:
                return None
            return result.model_copy(update={"prefix": piece[:index],
                                             "started": True})
        return StepResult(prefix=piece)

    @abstractmethod
    def check_value(self, piece: bytes) -> Optional[StepResult]:
        """Test whether a token may extend the value (opener excluded).

        Args:
            piece: Non-empty bytes of the candidate token.

        Returns:
            A StepResult if the token is allowed, None otherwise.
        """

    @abstractmethod
    def value(self) -> Any:
        """Return the Python value of the generated text."""

    def _forced_value(self) -> Optional[bytes]:
        """Return the only possible end of the value, if there is one."""
        return None

    def forced(self) -> Optional[StepResult]:
        """Return the only possible continuation, if there is one.

        When the remaining text is fully determined, the decoder appends it
        directly instead of asking the model, which saves model calls.

        Returns:
            A StepResult completing the value, or None.
        """
        rest = self._forced_value()
        if rest is None:
            return None
        if self.started:
            return StepResult(accepted=rest, done=True, started=True)
        opener = b" " + self.opener
        if not opener.startswith(self.prefix):
            opener = self.opener
        return StepResult(prefix=opener[len(self.prefix):], accepted=rest,
                          done=True, started=True)

    def is_complete(self) -> bool:
        """Tell whether the buffer already holds a complete value."""
        return False

    def commit(self, result: StepResult) -> None:
        """Append an accepted step to the generated text.

        Args:
            result: A StepResult previously returned by ``check``.
        """
        self.prefix += result.prefix
        self.started = self.started or result.started
        self.buffer += result.accepted
        self.state = result.state


class ChoiceConstraint(Constraint):
    """The value must be exactly one of a fixed list of strings.

    Used for function names and booleans.

    Attributes:
        choices: Allowed values (already JSON-escaped when needed).
        terminator: Bytes that must follow the value (``"`` for a name), so
            that a choice which is a prefix of another one stays valid.
    """

    choices: List[bytes] = Field(min_length=1)
    terminator: bytes = b""

    def _candidates(self) -> List[bytes]:
        """Return every choice followed by the terminator."""
        return [choice + self.terminator for choice in self.choices]

    def check_value(self, piece: bytes) -> Optional[StepResult]:
        """See :meth:`Constraint.check_value`."""
        text = self.buffer + piece
        candidates = self._candidates()
        if any(candidate.startswith(text) for candidate in candidates):
            return StepResult(accepted=piece, done=text in candidates)
        for candidate in candidates:
            if text.startswith(candidate):
                return StepResult(accepted=candidate[len(self.buffer):],
                                  done=True)
        return None

    def _forced_value(self) -> Optional[bytes]:
        """See :meth:`Constraint._forced_value`."""
        remaining = [candidate for candidate in self._candidates()
                     if candidate.startswith(self.buffer)]
        if len(remaining) == 1:
            return remaining[0][len(self.buffer):]
        return None

    def is_complete(self) -> bool:
        """See :meth:`Constraint.is_complete`."""
        return self.buffer in self._candidates()

    def value(self) -> str:
        """Return the selected choice without its terminator."""
        text = self.buffer
        if self.terminator and text.endswith(self.terminator):
            text = text[:-len(self.terminator)]
        return text.decode("utf-8")


class BooleanConstraint(ChoiceConstraint):
    """A JSON boolean: ``true`` or ``false``."""

    choices: List[bytes] = Field(default_factory=lambda: [b"true", b"false"])

    def value(self) -> bool:  # type: ignore[override]
        """Return the generated boolean."""
        return self.buffer == b"true"


class NumberConstraint(Constraint):
    """A JSON number (or integer), ended by ``,`` ``}`` or whitespace.

    Attributes:
        integer: Forbid fractional parts and exponents.
        max_length: Maximum number of characters of the number.
    """

    integer: bool = False
    max_length: int = 32

    def _is_prefix(self, text: bytes) -> bool:
        """Tell whether ``text`` can still become a valid number."""
        pattern = INTEGER_PREFIX if self.integer else NUMBER_PREFIX
        return pattern.fullmatch(text) is not None

    def _is_number(self, text: bytes) -> bool:
        """Tell whether ``text`` is a complete valid number."""
        pattern = INTEGER_COMPLETE if self.integer else NUMBER_COMPLETE
        return pattern.fullmatch(text) is not None

    def check_value(self, piece: bytes) -> Optional[StepResult]:
        """See :meth:`Constraint.check_value`."""
        for index, byte in enumerate(piece):
            text = self.buffer + piece[:index + 1]
            if len(text) <= self.max_length and self._is_prefix(text):
                continue
            head = self.buffer + piece[:index]
            if byte in b",} \n\t" and self._is_number(head):
                return StepResult(accepted=piece[:index], done=True)
            return None
        return StepResult(accepted=piece)

    def is_complete(self) -> bool:
        """See :meth:`Constraint.is_complete`."""
        return self.started and self._is_number(self.buffer)

    def value(self) -> float | int:
        """Return the generated number as a float (or int)."""
        if self.integer:
            return int(self.buffer)
        return float(self.buffer)


class StringConstraint(Constraint):
    """The content of a JSON string, ended by an unescaped quote.

    The model generates the opening quote, the escaped content, and decides
    where the string ends with the closing quote.

    Attributes:
        safe_length: Length of the longest buffer prefix that ends outside
            an escape sequence (used when generation must be cut short).
    """

    opener: bytes = b'"'
    safe_length: int = 0

    def check_value(self, piece: bytes) -> Optional[StepResult]:
        """See :meth:`Constraint.check_value`."""
        state = self.state
        for index, byte in enumerate(piece):
            if state == NORMAL:
                if byte == QUOTE:
                    return StepResult(accepted=piece[:index], done=True)
                if byte == BACKSLASH:
                    state = ESCAPE
                elif byte < 0x20:
                    return None
            elif state == ESCAPE:
                if byte in SIMPLE_ESCAPES:
                    state = NORMAL
                elif byte == ord("u"):
                    state = HEX_BASE + 4
                else:
                    return None
            else:
                if byte not in HEX_DIGITS:
                    return None
                state = state - 1 if state - 1 > HEX_BASE else NORMAL
        return StepResult(accepted=piece, state=state)

    def commit(self, result: StepResult) -> None:
        """See :meth:`Constraint.commit`."""
        super().commit(result)
        if self.state == NORMAL:
            self.safe_length = len(self.buffer)

    def is_complete(self) -> bool:
        """See :meth:`Constraint.is_complete`."""
        return self.started

    def value(self) -> str:
        """Return the decoded string content."""
        raw = self.buffer[:self.safe_length].decode("utf-8", "replace")
        text = json.loads('"' + raw + '"')
        if not isinstance(text, str):
            return ""
        return text.encode("utf-8", "replace").decode("utf-8", "replace")


def constraint_for(param_type: str) -> Constraint:
    """Create the constraint matching a parameter type.

    Args:
        param_type: One of ``number``, ``integer``, ``string``, ``boolean``.

    Returns:
        A fresh constraint instance.

    Raises:
        ValueError: If the type is not supported.
    """
    if param_type == "number":
        return NumberConstraint()
    if param_type == "integer":
        return NumberConstraint(integer=True)
    if param_type == "string":
        return StringConstraint()
    if param_type == "boolean":
        return BooleanConstraint()
    raise ValueError(f"unsupported parameter type '{param_type}'")
