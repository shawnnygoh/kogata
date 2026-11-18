import pytest
import torch

from kogata.attention import Head, MHAWrapper

DIM, HEAD_DIM = 32, 8


def test_head_output_shape():
    x = torch.randn(2, 5, DIM)
    assert Head(DIM, HEAD_DIM)(x).shape == (2, 5, HEAD_DIM)


def test_head_has_no_biases():
    names = [name for name, _ in Head(DIM, HEAD_DIM).named_parameters()]
    assert not any(name.endswith("bias") for name in names)


def test_state_dict_holds_only_projection_weights():
    keys = set(Head(DIM, HEAD_DIM).state_dict())
    assert keys == {"q_proj.weight", "k_proj.weight", "v_proj.weight"}


def test_attention_weights_are_lower_triangular():
    # With x = I and v_proj = I, v = I, so the output is the weight matrix itself
    T = HEAD_DIM
    head = Head(T, HEAD_DIM)
    with torch.no_grad():
        head.v_proj.weight.copy_(torch.eye(HEAD_DIM))
    x = torch.eye(T).unsqueeze(0)  # (1, T, T)
    weights = head(x)[0]  # (T, T)

    assert torch.all(weights.triu(diagonal=1) == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(T))


def test_first_token_attends_only_to_itself():
    head = Head(DIM, HEAD_DIM)
    x = torch.randn(2, 5, DIM)
    v = head.v_proj(x)
    assert torch.allclose(head(x)[:, 0], v[:, 0], atol=1e-6)


def test_changing_a_token_leaves_earlier_outputs_unchanged():
    torch.manual_seed(0)
    head = Head(DIM, HEAD_DIM)
    x = torch.randn(1, 8, DIM)
    t = 4
    x2 = x.clone()
    x2[0, t] = torch.randn(DIM)

    out, out2 = head(x), head(x2)
    assert torch.allclose(out[0, :t], out2[0, :t], atol=1e-6)
    assert not torch.allclose(out[0, t:], out2[0, t:], atol=1e-6)


def test_accepts_any_sequence_length():
    head = Head(DIM, HEAD_DIM)
    for T in (1, 7, 300):
        assert head(torch.randn(1, T, DIM)).shape == (1, T, HEAD_DIM)


def test_mask_is_built_on_the_input_device():
    # "meta" tensors have shapes but no data, so this checks devices without a GPU
    head = Head(DIM, HEAD_DIM).to("meta")
    x = torch.randn(2, 5, DIM, device="meta")
    assert head(x).device.type == "meta"


def test_matches_token_by_token_causal_attention():
    torch.manual_seed(0)
    head = Head(DIM, HEAD_DIM)
    x = torch.randn(1, 6, DIM)
    q, k, v = head.q_proj(x)[0], head.k_proj(x)[0], head.v_proj(x)[0]

    expected = []
    for i in range(6):
        # Token i only sees tokens 0..i
        scores = torch.stack([q[i] @ k[j] for j in range(i + 1)]) / HEAD_DIM**0.5
        weights = scores.softmax(dim=0)
        expected.append(sum(weights[j] * v[j] for j in range(i + 1)))

    assert torch.allclose(head(x)[0], torch.stack(expected), atol=1e-6)


N_HEADS = 4


def test_mha_wrapper_output_shape():
    x = torch.randn(2, 5, DIM)
    assert MHAWrapper(DIM, N_HEADS)(x).shape == (2, 5, DIM)


def test_mha_wrapper_stores_dim():
    # Blocks and models check that every component agrees on dim
    assert MHAWrapper(DIM, N_HEADS, head_dim=16).dim == DIM


def test_mha_wrapper_head_dim_defaults_to_dim_over_n_heads():
    attn = MHAWrapper(DIM, N_HEADS)
    assert len(attn.heads) == N_HEADS
    assert all(head.head_dim == DIM // N_HEADS for head in attn.heads)


def test_mha_wrapper_accepts_explicit_head_dim():
    # n_heads * head_dim = 64, which need not equal dim
    attn = MHAWrapper(DIM, N_HEADS, head_dim=16)
    assert all(head.head_dim == 16 for head in attn.heads)
    assert attn(torch.randn(2, 5, DIM)).shape == (2, 5, DIM)


def test_mha_wrapper_rejects_dim_not_divisible_by_n_heads():
    with pytest.raises(ValueError):
        MHAWrapper(DIM, n_heads=5)


def test_mha_wrapper_has_no_biases():
    attn = MHAWrapper(DIM, N_HEADS)
    names = [name for name, _ in attn.named_parameters()]
    assert not any(name.endswith("bias") for name in names)


def test_mha_wrapper_parameter_count():
    # q, k, v for every head, plus o_proj: 4 * dim * (n_heads * head_dim)
    attn = MHAWrapper(DIM, N_HEADS, head_dim=16)
    assert sum(p.numel() for p in attn.parameters()) == 4 * DIM * N_HEADS * 16


def test_mha_wrapper_heads_have_independent_weights():
    attn = MHAWrapper(DIM, N_HEADS)
    first, second = attn.heads[0].q_proj.weight, attn.heads[1].q_proj.weight
    assert first is not second
    assert not torch.equal(first, second)


def test_mha_wrapper_concatenates_heads_in_order_then_projects():
    torch.manual_seed(0)
    attn = MHAWrapper(DIM, N_HEADS)
    x = torch.randn(2, 5, DIM)
    concat = torch.cat([head(x) for head in attn.heads], dim=-1)
    assert torch.allclose(attn(x), attn.o_proj(concat), atol=1e-6)


def test_mha_wrapper_is_causal():
    torch.manual_seed(0)
    attn = MHAWrapper(DIM, N_HEADS)
    x = torch.randn(1, 8, DIM)
    t = 4
    x2 = x.clone()
    x2[0, t] = torch.randn(DIM)

    out, out2 = attn(x), attn(x2)
    assert torch.allclose(out[0, :t], out2[0, :t], atol=1e-6)
    assert not torch.allclose(out[0, t:], out2[0, t:], atol=1e-6)


def test_mha_wrapper_state_dict_holds_every_head():
    # Heads must be registered submodules, or checkpoints silently drop them
    keys = set(MHAWrapper(DIM, N_HEADS).state_dict())
    expected = {
        f"heads.{i}.{name}_proj.weight" for i in range(N_HEADS) for name in "qkv"
    }
    assert keys == expected | {"o_proj.weight"}
