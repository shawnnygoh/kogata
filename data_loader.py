"""Dataset that slices a token sequence into input-target windows."""

import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Yields (x, y) pairs where y is x shifted one token to the right.

    Window i starts at token i * stride. With stride=1 the windows overlap
    almost entirely; with stride=block_size they do not overlap at all.
    """

    def __init__(self, token_ids: list[int], block_size: int, stride: int = 1):
        if block_size < 1 or stride < 1:
            raise ValueError("block_size and stride must be at least 1")
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.stride = stride

    def __len__(self) -> int:
        # Each window needs block_size inputs plus one extra token for the target
        last_start = len(self.data) - self.block_size - 1
        if last_start < 0:
            return 0
        return last_start // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for {len(self)} windows")
        start = idx * self.stride
        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]
        return x, y


if __name__ == "__main__":
    from tokenizer import CharTokenizer

    text = "Building a small language model from scratch is awesome!"
    tokenizer = CharTokenizer(text)
    dataset = TextDataset(tokenizer.encode(text), block_size=8, stride=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    x, y = next(iter(loader))
    print(f"Windows: {len(dataset)}, x: {tuple(x.shape)}, y: {tuple(y.shape)}")
    for xi, yi in zip(x, y):
        print(f"{tokenizer.decode(xi.tolist())!r} -> {tokenizer.decode(yi.tolist())!r}")
