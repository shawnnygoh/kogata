"""Building blocks for recipes."""

from torch.nn import Embedding, Linear

from kogata.attention import MHAWrapper

__all__ = ["Embedding", "Linear", "MHAWrapper"]
