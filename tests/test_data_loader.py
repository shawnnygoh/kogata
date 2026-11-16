import pytest
import torch
from torch.utils.data import DataLoader

from data_loader import TextDataset

TOKENS = list(range(20))


def test_target_is_input_shifted_by_one():
    dataset = TextDataset(TOKENS, block_size=4)
    for x, y in dataset:
        assert torch.equal(x[1:], y[:-1])
        assert y[-1] == x[-1] + 1


def test_length_with_stride_one():
    assert len(TextDataset(TOKENS, block_size=4)) == 16


def test_stride_spaces_out_windows():
    dataset = TextDataset(TOKENS, block_size=4, stride=4)
    assert len(dataset) == 4
    assert dataset[1][0].tolist() == [4, 5, 6, 7]


def test_last_window_stays_in_bounds():
    dataset = TextDataset(TOKENS, block_size=4, stride=3)
    x, y = dataset[len(dataset) - 1]
    assert len(x) == len(y) == 4
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_too_short_sequence_is_empty():
    assert len(TextDataset([1, 2, 3], block_size=4)) == 0


def test_batches_have_expected_shape():
    loader = DataLoader(TextDataset(TOKENS, block_size=4), batch_size=8)
    x, y = next(iter(loader))
    assert x.shape == y.shape == (8, 4)
    assert x.dtype == torch.long
