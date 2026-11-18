import torch

from kogata import nn
from kogata.attention import MHAWrapper


def test_reexports_kogata_components():
    assert nn.MHAWrapper is MHAWrapper


def test_reexports_torch_building_blocks():
    assert nn.Embedding is torch.nn.Embedding
    assert nn.Linear is torch.nn.Linear


def test_all_lists_every_public_name():
    public = {name for name in vars(nn) if not name.startswith("_")}
    assert set(nn.__all__) == public
