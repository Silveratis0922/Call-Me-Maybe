"""Unit tests for the constraints and the decoder (no real LLM needed)."""

import json
import random
from pathlib import Path
from typing import Dict, List

import pytest

from src.constraints import (BooleanConstraint, ChoiceConstraint,
                             NumberConstraint, StringConstraint)
from src.decoder import ConstrainedDecoder
from src.io_utils import load_functions, load_prompts
from src.models import FunctionDefinition
from src.vocabulary import Vocabulary, byte_level_alphabet


def feed(constraint: ChoiceConstraint | NumberConstraint | StringConstraint,
         pieces: List[bytes]) -> bool:
    """Feed tokens to a constraint; return False on the first rejection."""
    for piece in pieces:
        result = constraint.check(piece)
        if result is None:
            return False
        constraint.commit(result)
        if result.done:
            return True
    return True


def test_number_accepts_valid_numbers() -> None:
    constraint = NumberConstraint()
    assert feed(constraint, [b"-", b"12", b".5", b","])
    assert constraint.value() == -12.5


def test_number_rejects_invalid_tokens() -> None:
    constraint = NumberConstraint()
    assert constraint.check(b"a") is None
    assert constraint.check(b"01") is None
    constraint.commit(NumberConstraint().check(b"1") or pytest.fail())
    assert constraint.check(b".") is not None
    assert constraint.check(b"}") is not None
    assert constraint.check(b'"') is None


def test_number_needs_digits_before_terminator() -> None:
    constraint = NumberConstraint()
    constraint.commit(constraint.check(b"-") or pytest.fail())
    assert constraint.check(b",") is None


def test_integer_forbids_fraction() -> None:
    constraint = NumberConstraint(integer=True)
    constraint.commit(constraint.check(b"42") or pytest.fail())
    assert constraint.check(b".") is None
    assert feed(constraint, [b"}"])
    assert constraint.value() == 42


def test_string_handles_escapes_and_quote() -> None:
    constraint = StringConstraint()
    assert feed(constraint, [b' "a\\\"b', b"\\\\d+", b'",'])
    assert constraint.value() == 'a"b\\d+'


def test_string_rejects_bad_escape_and_control() -> None:
    assert StringConstraint().check(b'"\\q') is None
    assert StringConstraint().check(b'"a\nb') is None
    assert StringConstraint().check(b'"\\u12G4') is None
    assert StringConstraint().check(b"abc") is None


def test_string_empty() -> None:
    constraint = StringConstraint()
    assert feed(constraint, [b' ', b'"', b'"'])
    assert constraint.value() == ""


def test_choice_prefix_names() -> None:
    constraint = ChoiceConstraint(choices=[b"fn_a", b"fn_ab"],
                                  opener=b'"', terminator=b'"')
    assert feed(constraint, [b' "', b"fn", b"_a", b'",'])
    assert constraint.value() == "fn_a"
    assert ChoiceConstraint(choices=[b"x"]).check(b"y") is None


def test_choice_forced_completion() -> None:
    constraint = ChoiceConstraint(choices=[b"fn_add", b"fn_greet"],
                                  opener=b'"', terminator=b'"')
    assert constraint.forced() is None
    constraint.commit(constraint.check(b' "fn_g') or pytest.fail())
    forced = constraint.forced()
    assert forced is not None and forced.accepted == b'reet"'
    single = ChoiceConstraint(choices=[b"fn"], opener=b'"')
    forced = single.forced()
    assert forced is not None and forced.prefix == b' "'


def test_leading_space_is_optional() -> None:
    constraint = NumberConstraint()
    assert feed(constraint, [b" -", b"5", b"}"])
    assert constraint.value() == -5.0
    assert NumberConstraint().check(b"  5") is None
    assert BooleanConstraint().check(b" true") is not None


def test_boolean() -> None:
    constraint = BooleanConstraint()
    assert feed(constraint, [b"fal", b"se"])
    assert constraint.value() is False


def test_byte_level_alphabet_is_bijective() -> None:
    alphabet = byte_level_alphabet()
    assert len(alphabet) == 256
    assert sorted(alphabet.values()) == list(range(256))
    assert alphabet["Ġ"] == ord(" ")


class FakeTensor:
    """Minimal stand-in for a torch tensor."""

    def __init__(self, ids: List[int]) -> None:
        self.ids = ids

    def tolist(self) -> List[List[int]]:
        return [self.ids]


class RandomModel:
    """Fake LLM with one token per byte and random logits."""

    def __init__(self, vocab_path: str, seed: int) -> None:
        self.vocab_path = vocab_path
        self.rng = random.Random(seed)

    def encode(self, text: str) -> FakeTensor:
        return FakeTensor(list(text.encode("utf-8")))

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        return [self.rng.gauss(0, 1) for _ in range(256 + 8)]

    def get_path_to_vocab_file(self) -> str:
        return self.vocab_path


@pytest.fixture()
def vocab_file(tmp_path: Path) -> str:
    reverse: Dict[int, str] = {byte: char for char, byte
                               in byte_level_alphabet().items()}
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({reverse[i]: i for i in range(256)}))
    return str(path)


def test_vocabulary_maps_bytes(vocab_file: str) -> None:
    vocab = Vocabulary(path=vocab_file)
    vocab.load(260)
    assert vocab.token_bytes(ord(" ")) == b" "
    assert vocab.token_bytes(259) is None


@pytest.mark.parametrize("seed", range(20))
def test_random_model_always_gives_valid_calls(vocab_file: str,
                                               seed: int) -> None:
    functions = [
        FunctionDefinition.model_validate(item) for item in [
            {"name": "fn_add", "parameters": {"a": {"type": "number"},
                                              "b": {"type": "integer"}}},
            {"name": "fn_add_text", "parameters": {"s": {"type": "string"},
                                                   "f": {"type": "bool"}}},
            {"name": "fn_none", "parameters": {}},
        ]
    ]
    decoder = ConstrainedDecoder(model=RandomModel(vocab_file, seed),
                                 functions=functions, max_value_tokens=12)
    call = decoder.call("anything")
    function = next(f for f in functions if f.name == call.name)
    assert list(call.parameters) == list(function.parameters)
    python_types = {"number": float, "integer": int, "string": str,
                    "boolean": bool}
    for key, spec in function.parameters.items():
        assert type(call.parameters[key]) is python_types[spec.type]
    json.loads(json.dumps(call.model_dump()))


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_functions(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("[{,}]")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_prompts(str(bad))
    partial = tmp_path / "partial.json"
    partial.write_text('[{"prompt": "ok"}, {"nope": 1}, 3]')
    assert [p.prompt for p in load_prompts(str(partial))] == ["ok"]
    wrong_type = tmp_path / "fns.json"
    wrong_type.write_text('[{"name": "f", "parameters": '
                          '{"x": {"type": "matrix"}}}]')
    with pytest.raises(ValueError, match="no valid function"):
        load_functions(str(wrong_type))
