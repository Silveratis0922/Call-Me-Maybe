"""Constrained decoding of function calls with a small LLM."""

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.constraints import (ChoiceConstraint, Constraint, StepResult,
                             StringConstraint, constraint_for)
from src.models import FunctionCall, FunctionDefinition
from src.prompt import build_prompt
from src.vocabulary import Vocabulary


class ConstrainedDecoder(BaseModel):
    """Generate schema-valid function calls token by token.

    The JSON skeleton (braces, keys, quotes, commas) is written by the
    decoder itself. The model only generates the free parts: the function
    name and the argument values, each under a constraint that masks every
    token which would break the JSON syntax or the expected type.

    Attributes:
        model: A ``Small_LLM_Model`` instance from ``llm_sdk``.
        functions: The functions the model may call.
        max_value_tokens: Maximum number of generated tokens per value.
        verbose: Print every generation step on stderr.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    functions: List[FunctionDefinition]
    max_value_tokens: int = 64
    verbose: bool = False
    _vocab: Optional[Vocabulary] = PrivateAttr(default=None)
    _model_calls: int = PrivateAttr(default=0)

    @property
    def model_calls(self) -> int:
        """Number of forward passes done since the decoder was created."""
        return self._model_calls

    def _vocabulary(self, size: int) -> Vocabulary:
        """Load the vocabulary the first time it is needed.

        Args:
            size: Number of logits returned by the model.

        Returns:
            The token table.
        """
        if self._vocab is None:
            path = str(self.model.get_path_to_vocab_file())
            vocab = Vocabulary(path=path)
            vocab.load(size)
            self._vocab = vocab
        return self._vocab

    def _encode(self, text: str) -> List[int]:
        """Tokenize text into a flat list of ids.

        Args:
            text: Text to encode.

        Returns:
            The token ids.
        """
        if not text:
            return []
        encoded = self.model.encode(text)
        rows = encoded.tolist()
        flat = rows[0] if rows and isinstance(rows[0], list) else rows
        return [int(token) for token in flat]

    def _log(self, message: str) -> None:
        """Print a debug message when verbose mode is on.

        Args:
            message: Text to display.
        """
        if self.verbose:
            print(message, file=sys.stderr)

    def _next_token(self, ids: List[int], constraint: Constraint,
                    must_end: bool = False
                    ) -> Tuple[int, bytes, StepResult]:
        """Pick the most likely token allowed by the constraint.

        Conceptually every forbidden token gets a logit of minus infinity
        and the arg-max of the masked logits is taken. Checking the tokens
        from the highest logit downwards and stopping at the first allowed
        one gives exactly the same token without testing the whole
        vocabulary at every step.

        Args:
            ids: Token ids of the text so far.
            constraint: The constraint of the value being generated.
            must_end: Only allow tokens that leave a complete value (used
                for the last token allowed for a value).

        Returns:
            The chosen token id, its bytes and the constraint check result.

        Raises:
            RuntimeError: If no token of the vocabulary is allowed.
        """
        raw_logits = self.model.get_logits_from_input_ids(ids)
        self._model_calls += 1
        logits = np.asarray(raw_logits, dtype=np.float32)
        vocab = self._vocabulary(int(logits.shape[0]))
        rejected = 0
        for token_id in np.argsort(-logits, kind="stable"):
            piece = vocab.token_bytes(int(token_id))
            if piece is None:
                rejected += 1
                continue
            result = constraint.check(piece)
            if result is not None and must_end and not result.done:
                trial = constraint.model_copy()
                trial.commit(result)
                if not trial.is_complete():
                    result = None
            if result is not None:
                self._log(f"    token {piece!r:<16} "
                          f"(logit {float(logits[token_id]):6.2f}, "
                          f"{rejected} better tokens masked)")
                return int(token_id), piece, result
            rejected += 1
        raise RuntimeError("no token satisfies the constraint")

    def _generate(self, ids: List[int], constraint: Constraint) -> List[int]:
        """Generate one constrained value.

        Args:
            ids: Token ids of the text so far.
            constraint: The constraint of the value (updated in place).

        Returns:
            The token ids extended with the generated value.

        Raises:
            RuntimeError: If the value cannot be completed.
        """
        for step in range(self.max_value_tokens):
            forced = constraint.forced()
            if forced is not None:
                text = forced.prefix + forced.accepted
                self._log(f"    forced {text!r}")
                ids = ids + self._encode(text.decode("utf-8"))
                constraint.commit(forced)
                return ids
            last = step == self.max_value_tokens - 1
            token_id, piece, result = self._next_token(ids, constraint,
                                                       must_end=last)
            kept = result.prefix + result.accepted
            if kept == piece:
                ids = ids + [token_id]
            else:
                ids = ids + self._encode(kept.decode("utf-8", "replace"))
            constraint.commit(result)
            if result.done:
                return ids
        if constraint.is_complete():
            self._log("    token limit reached, value cut")
            return ids
        raise RuntimeError("value too long, generation stopped")

    def call(self, request: str) -> FunctionCall:
        """Translate one natural-language request into a function call.

        Args:
            request: The user prompt.

        Returns:
            The function call, valid against the function definitions.

        Raises:
            RuntimeError: If generation fails.
        """
        start = time.perf_counter()
        self._log(f"\n>>> {request}")
        ids = self._encode(build_prompt(self.functions, request))
        ids += self._encode('{"name":')

        by_name = {function.name: function for function in self.functions}
        names = [json.dumps(name)[1:-1].encode("utf-8") for name in by_name]
        self._log("  [name]")
        name_constraint = ChoiceConstraint(choices=names, opener=b'"',
                                           terminator=b'"')
        ids = self._generate(ids, name_constraint)
        function = by_name[json.loads(f'"{name_constraint.value()}"')]

        ids += self._encode(', "parameters": {')
        parameters: Dict[str, Any] = {}
        for index, (key, spec) in enumerate(function.parameters.items()):
            separator = ", " if index else ""
            ids += self._encode(f"{separator}{json.dumps(key)}:")
            self._log(f"  [{key}: {spec.type}]")
            constraint = constraint_for(spec.type)
            ids = self._generate(ids, constraint)
            if isinstance(constraint, StringConstraint):
                ids += self._encode('"')
            parameters[key] = constraint.value()

        self._log(f"  done in {time.perf_counter() - start:.1f}s")
        return FunctionCall(prompt=request, name=function.name,
                            parameters=parameters)
