*This project has been created as part of the 42 curriculum by tchantro.*

# Call Me Maybe

## Description

**Call Me Maybe** is an introduction to *function calling* with Large
Language Models. Given a natural-language request such as
`"What is the sum of 2 and 3?"`, the program does not answer `5`: it
produces the function call that solves it:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The model used is **Qwen/Qwen3-0.6B**, a very small model which, when simply
prompted, often produces broken or wrongly typed JSON. This project uses
**constrained decoding**: at every generation step, the tokens that would
break the JSON syntax or the function schema are masked, so the output is
**always** valid JSON matching `functions_definition.json`, whatever the model
"wants" to write.

No high-level generation library is used (no transformers generation,
outlines, dspy...): the program only uses the `llm_sdk` wrapper to get raw
logits, the vocabulary file and the tokenizer, and implements the decoding
loop itself with `numpy` and `pydantic`.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- About 1.5 GB of disk space for the model (downloaded automatically from
  Hugging Face the first time).

### Installation

```bash
make install        # same as: uv sync
```

`llm_sdk/` is declared as a local path dependency in `pyproject.toml`, so
`uv sync` installs it together with its own dependencies.

### Execution

```bash
make run            # uv run python -m src
```

Full command with every option:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

| Option | Default | Meaning |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | available functions |
| `--input` | `data/input/function_calling_tests.json` | prompts to process |
| `--output` | `data/output/function_calling_results.json` | results file |
| `--model` | `Qwen/Qwen3-0.6B` | any causal LM of the Hugging Face Hub with a byte-level BPE vocabulary |
| `--verbose` | off | print every generation step (chosen tokens, masked tokens) |

### Other Makefile rules

| Rule | Action |
|---|---|
| `make debug` | run the program under `pdb` |
| `make lint` | `flake8` + `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8` + `mypy --strict` |
| `make test` | run the unit tests with `pytest` |
| `make clean` | remove caches (`__pycache__`, `.mypy_cache`, ...) |

## Example usage

```console
$ uv run python -m src
Loading model Qwen/Qwen3-0.6B...
[1/11] 'What is the sum of 2 and 3?' -> fn_add_numbers({'a': 2.0, 'b': 3.0})
[2/11] 'What is the sum of 265 and 345?' -> fn_add_numbers({'a': 265.0, 'b': 345.0})
[3/11] 'Greet shrek' -> fn_greet({'name': 'shrek'})
...
11/11 prompts processed in 103.0s (128 model calls). Results written to data/output/function_calling_results.json
```

With `--verbose`, the generation process is shown token by token:

```text
>>> What is the sum of 2 and 3?
  [name]
    token b' "'            (logit  36.15, 0 better tokens masked)
    token b'fn'            (logit  41.67, 0 better tokens masked)
    token b'_add'          (logit  35.72, 0 better tokens masked)
    forced b'_numbers"'
  [a: number]
    token b' '             (logit  32.73, 0 better tokens masked)
    token b'2'             (logit  38.24, 0 better tokens masked)
    token b','             (logit  30.93, 0 better tokens masked)
  [b: number]
    token b' '             (logit  36.81, 0 better tokens masked)
    token b'3'             (logit  39.23, 0 better tokens masked)
    token b'}}'            (logit  30.06, 0 better tokens masked)
  done in 7.1s
```

## Algorithm explanation

### 1. The generation loop

An LLM generates text one token at a time:

```
prompt -> token ids -> LLM -> logits (one score per vocabulary token) -> next token -> repeat
```

`llm_sdk` only gives the raw logits of the next token
(`get_logits_from_input_ids`). The loop, the token selection and the
stopping rules are implemented in [`src/decoder.py`](src/decoder.py).

### 2. Reading the vocabulary

To know whether a token is allowed, we must know which text it represents.
`get_path_to_vocab_file()` returns Qwen's `vocab.json` (token string -> id).
Qwen uses *byte-level BPE*: tokens are stored with a printable alphabet
where every byte is replaced by a unicode character (a space becomes `Ġ`,
a newline `Ċ`...). [`src/vocabulary.py`](src/vocabulary.py) rebuilds this
alphabet (the GPT-2 `bytes_to_unicode` table), inverts it, and converts every
token into its **raw bytes**. All constraints then work on bytes, which also
handles tokens that contain only part of a multi-byte UTF-8 character.
Special tokens (`<|im_end|>`...) are absent from `vocab.json`: they are
never allowed.

### 3. The decoder writes the skeleton, the model fills the holes

The structure of the answer is known in advance, so the decoder writes it
directly, without asking the model:

```
{"name":<NAME>, "parameters": {"a":<NUMBER>, "b":<NUMBER>}}
        ^^^^^^ generated          ^^^^^^^^      ^^^^^^^^ generated
```

Everything else (braces, keys, colons, commas) is *forced* text appended to
the token sequence. Each generated hole starts with an optional space and,
for strings and names, the opening quote: these are generated by the model
too (see *Challenges faced*). The model only generates:

1. **the function name**: the LLM chooses the function, but only among the
   names of `functions_definition.json`;
2. **each argument value**, in the order of the definition, under the
   constraint of its type.

### 4. Constraints

Each hole has a constraint ([`src/constraints.py`](src/constraints.py)).
For a candidate token, `check(token_bytes)` answers "can this token extend
the current value and still lead to a valid value?":

| Type | Constraint |
|---|---|
| function name | `ChoiceConstraint`: `buffer + token` must be a prefix of one of the names followed by `"`. The final quote makes `fn_add` and `fn_add_numbers` distinguishable. |
| `number` / `integer` | `NumberConstraint`: the text must stay a prefix of the JSON number grammar `-?(0\|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?` (no fraction or exponent for integers). The number ends when the model produces `,`, `}` or a space after a complete number. |
| `string` | `StringConstraint`: a small state machine over the bytes of a JSON string: control characters are forbidden, only valid escapes are accepted (`\"`, `\\`, `\n`, `\uXXXX`...), and an unescaped `"` ends the value. |
| `boolean` | `BooleanConstraint`: `true` or `false`. |

A token may contain the end of the value and more (for example `3,` or
`",`). In that case only the part belonging to the value is kept, and the
token ids are re-encoded from the kept text so that the context given to the
model always matches the forced skeleton.

### 5. Masking

At each step:

1. the model returns the logits of all ~150 000 tokens;
2. every token that the constraint rejects gets a logit of **−∞**;
3. the token with the highest remaining logit is chosen (greedy decoding).

Instead of testing the whole vocabulary, the decoder sorts the logits
(`numpy.argsort`) and tests the tokens from the best downwards, keeping the
first allowed one. This is **exactly** the arg-max of the masked logits, but
it usually stops after a handful of checks instead of 150 000.

Because every chosen token is allowed by the constraint, the produced value
is always valid: the JSON output is guaranteed parseable and schema-compliant
by construction, not by luck.

### 6. Forced completion and stopping

- When only one choice is still possible (for example `fn_gr` can only be
  `fn_greet`), the rest is appended directly without calling the model.
- Each value is limited to `max_value_tokens` (64). On the last allowed step
  only tokens leaving a *complete* value are accepted, and strings are cut at
  the last position outside an escape sequence, so generation always ends
  with a valid value.

## Design decisions

- **The model chooses the function.** The name is generated by the LLM
  under a constraint; there is no keyword matching or heuristic.
- **Skeleton forced by the program.** Generating braces and keys with the
  model would only cost model calls and add failure modes. Writing them
  ourselves guarantees the exact output keys (`prompt`, `name`,
  `parameters`) and the order of the arguments.
- **Byte-level constraints.** Working on bytes rather than decoded strings
  avoids errors with tokens holding partial UTF-8 characters.
- **Compact prompt.** `llm_sdk` has no KV cache: every call re-processes the
  whole prompt. The function catalogue is therefore rendered as one short
  line per function (`- fn_add_numbers(a: number, b: number) -> number: ...`)
  in the Qwen chat format, with thinking disabled
  (`<think>\n\n</think>`), to keep each forward pass cheap.
- **Lazy arg-max** instead of building a full mask (see above): same result,
  much faster.
- **pydantic everywhere.** Inputs (`FunctionDefinition`, `PromptItem`),
  outputs (`FunctionCall`), CLI arguments and all classes (decoder,
  constraints, vocabulary) are pydantic models. Invalid entries of the input
  files are reported and skipped instead of stopping the program. Type
  aliases (`int`, `float`, `str`, `bool`) are accepted.
- **Numbers as floats.** The `number` type is returned as a Python `float`
  (`2.0`), as in the subject example; `integer` gives an `int`.
- **No hard-coded behaviour.** Nothing depends on the example files: any
  function list with `number`, `integer`, `string` and `boolean` parameters
  works.

## Performance analysis

Measured on the provided examples, CPU only (10 cores, no GPU):

| Metric | Result |
|---|---|
| Valid JSON | 100 % (guaranteed by construction) |
| Schema compliance | 100 % (names, keys and types always match) |
| Correct function | 11 / 11 |
| Correct arguments | 11 / 11 |
| Total time | ~100 s for 11 prompts, ~130 model calls (model loading excluded) |

Extra edge cases tested with custom files (booleans, integers, negative
floats, large numbers, accents and quotes in strings, empty request, invalid
entries): every output is valid; the only imperfection seen is the model
dropping inner quotes of a name (`Zoë "the great" O'Neil` ->
`Zoë the great O'Neil`), which is a model choice, not a format error.

Most of the time is spent in the forward passes of the model. Speed comes
from: forced skeleton and forced name completion (fewer model calls), a
compact prompt (cheaper calls), and lazy masking (negligible decoding
overhead). The program prints the number of model calls at the end.

## Challenges faced

- **Tokens are not characters.** A single token can be `",` or `3}`: it can
  close a value and start the skeleton. The constraints return the accepted
  part of the token, and the token ids are rebuilt from the text actually
  kept.
- **Byte-level vocabulary.** `vocab.json` does not contain readable text;
  the GPT-2 byte alphabet had to be reimplemented to recover the bytes.
- **Prefix names.** With `fn_add` and `fn_add_numbers`, the name `fn_add`
  is complete but could still continue; adding the closing quote to every
  choice makes the choice unambiguous.
- **Forcing the space breaks the tokenization.** The first version forced
  `"y": ` (with a space) before a value. For `Multiply ... by -0.5` the
  model then produced `0.5`: Qwen normally writes ` -` as one token, and
  after a forced space the best token was a digit. Letting the model
  generate the space and the opening quote itself (the `opener` of the
  constraints) fixed it.
- **Small model ambiguity.** For `Replace all numbers ... with NUMBERS`,
  after the opening quote the model hesitated between `NUM` and `([`. A
  generic instruction in the system prompt ("when an argument is a pattern
  or a regex, write the regular expression...") was enough to get
  `([0-9]+)`.
- **Escapes in strings.** The regex prompt (`Replace all numbers ... with
  NUMBERS`) can require backslashes (`"\\d+"`) inside a JSON string; the
  string state machine accepts only valid JSON escapes, and the value is
  decoded with `json.loads`.
- **Speed without KV cache.** Every model call costs a full forward pass, so
  avoiding calls (forcing) and keeping the prompt short mattered more than
  micro-optimisations.

## Testing strategy

- **Unit tests** (`make test`, [`tests/`](tests/)) for each constraint:
  valid and invalid numbers, integers, escapes, control characters, empty
  strings, prefix names, forced completion, booleans, and the byte alphabet.
- **Random model test**: a fake LLM returning *random logits* is plugged into
  the real decoder, 20 times with different seeds. Even with a model that
  "says anything", every result has a known function name, exactly the
  expected keys, and values of the right Python type. This checks the
  100 % validity guarantee independently of the real model.
- **Error handling tests**: missing file, invalid JSON, invalid entries
  skipped, unsupported parameter types.
- **End-to-end runs** on the provided files and on custom files (special
  characters, large numbers, functions with several parameters, ambiguous
  prompts), checking the output with `json.load`.
- `make lint` (flake8 + mypy) must report no issue.

## Resources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face — Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [OpenAI GPT-2 `encoder.py` (`bytes_to_unicode`)](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)](https://arxiv.org/abs/2307.09702)
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

### Use of AI

An AI assistant (Claude) was used to read and summarise the subject, to
inspect the provided `llm_sdk` package, to help write the project skeleton
(`pyproject.toml`, Makefile), the constraint state machines, the decoder,
the unit tests and this README. Every part was reviewed, tested and is
understood by the author, who can explain and modify it.
