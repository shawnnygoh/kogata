"""Character-level tokenizer: one token per unique character."""

import json
from pathlib import Path


class CharTokenizer:
    """Maps each unique character in a corpus to an integer ID and back.

    IDs follow the sorted order of the characters, so the same corpus always
    gives the same IDs.

    Args:
        text: Corpus whose characters make up the vocabulary.
    """

    def __init__(self, text: str):
        # Sort so the same corpus always produces the same IDs
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        """Convert a string into a list of token IDs.

        Raises:
            ValueError: If `s` contains a character not in the vocabulary.
        """
        try:
            return [self.stoi[ch] for ch in s]
        except KeyError as e:
            ch = e.args[0]
            raise ValueError(f"Character {ch!r} is not in the vocabulary") from None

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back into a string."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        """Write the vocabulary to a JSON file."""
        chars = [self.itos[i] for i in range(self.vocab_size)]
        Path(path).write_text(json.dumps(chars, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Rebuild a tokenizer from a file written by `save`."""
        chars = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls("".join(chars))


if __name__ == "__main__":
    text = "Building a small language model from scratch is awesome!"
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Encoded (first 15): {ids[:15]}")
    print(f"Decoded: {tokenizer.decode(ids)}")
