import torch

from model import Head

DIM, HEAD_DIM = 32, 8


def test_head_output_shape():
    x = torch.randn(2, 5, DIM)
    assert Head(DIM, HEAD_DIM)(x).shape == (2, 5, HEAD_DIM)


def test_head_has_no_biases():
    names = [name for name, _ in Head(DIM, HEAD_DIM).named_parameters()]
    assert not any(name.endswith("bias") for name in names)


def test_identical_tokens_give_identical_outputs():
    # With no mask and no positions, every token attends to the same set
    x = torch.randn(1, 1, DIM).expand(1, 4, DIM)
    out = Head(DIM, HEAD_DIM)(x)
    assert torch.allclose(out[0, 0], out[0, 3])


def test_matches_token_by_token_attention():
    torch.manual_seed(0)
    head = Head(DIM, HEAD_DIM)
    x = torch.randn(1, 6, DIM)
    q, k, v = head.q_proj(x)[0], head.k_proj(x)[0], head.v_proj(x)[0]

    expected = []
    for i in range(6):
        scores = torch.stack([q[i] @ k[j] for j in range(6)]) / HEAD_DIM**0.5
        weights = scores.softmax(dim=0)
        expected.append(sum(weights[j] * v[j] for j in range(6)))

    assert torch.allclose(head(x)[0], torch.stack(expected), atol=1e-6)
