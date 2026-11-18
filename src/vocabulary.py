"""Mapping between token ids and the raw bytes they represent."""

import json
from typing import Dict, List, Optional

from pydantic import BaseModel, PrivateAttr


def byte_level_alphabet() -> Dict[str, int]:
    """Build the byte-level BPE alphabet used by GPT-2 style tokenizers.

    Byte-level BPE vocabularies never contain raw bytes: each byte is
    replaced by a printable unicode character (for instance a space becomes
    ``Ġ`` and a newline ``Ċ``). This function returns the reverse mapping,
    from that printable character back to the original byte.

    Returns:
        A dictionary mapping each alphabet character to its byte value.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + shift)
            shift += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


class Vocabulary(BaseModel):
    """Token id to bytes lookup table built from ``vocab.json``.

    Attributes:
        path: Location of the vocabulary file.
    """

    path: str
    _tokens: List[Optional[bytes]] = PrivateAttr(default_factory=list)

    def load(self, size: int) -> None:
        """Read the vocabulary file and decode every token.

        Args:
            size: Number of logits produced by the model; ids without an
                entry in the file (special or padding tokens) map to None.

        Raises:
            ValueError: If the file cannot be read or is malformed.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load vocabulary '{self.path}': {exc}") from None
        if not isinstance(raw, dict):
            raise ValueError(f"vocabulary '{self.path}' is not an object")
        alphabet = byte_level_alphabet()
        tokens: List[Optional[bytes]] = [None] * size
        for text, token_id in raw.items():
            if not isinstance(token_id, int) or not 0 <= token_id < size:
                continue
            try:
                tokens[token_id] = bytes(alphabet[char] for char in text)
            except KeyError:
                tokens[token_id] = text.encode("utf-8")
        self._tokens = tokens

    def __len__(self) -> int:
        """Return the number of token ids known by the table."""
        return len(self._tokens)

    def token_bytes(self, token_id: int) -> Optional[bytes]:
        """Return the bytes of a token.

        Args:
            token_id: The token id.

        Returns:
            The raw bytes, or None for special and unknown tokens.
        """
        if 0 <= token_id < len(self._tokens):
            return self._tokens[token_id]
        return None
