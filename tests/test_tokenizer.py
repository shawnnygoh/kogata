import pytest

from tokenizer import CharTokenizer

TEXT = "hello world, héllo wörld! 你好"


def test_round_trip():
    tokenizer = CharTokenizer(TEXT)
    assert tokenizer.decode(tokenizer.encode(TEXT)) == TEXT


def test_vocab_size_counts_unique_characters():
    assert CharTokenizer("aabbc").vocab_size == 3


def test_unknown_character_raises():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(ValueError, match="'z'"):
        tokenizer.encode("abz")


def test_save_and_load(tmp_path):
    tokenizer = CharTokenizer(TEXT)
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.stoi == tokenizer.stoi
    assert loaded.encode(TEXT) == tokenizer.encode(TEXT)
