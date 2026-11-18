import inspect

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


def test_kogata_components_document_args_and_shape():
    exports = [getattr(nn, name) for name in nn.__all__]
    components = [obj for obj in exports if obj.__module__.startswith("kogata.")]
    assert components
    for cls in components:
        doc = inspect.getdoc(cls) or ""
        assert "\nArgs:\n" in doc, cls.__name__
        assert "\nShape:\n" in doc, cls.__name__
