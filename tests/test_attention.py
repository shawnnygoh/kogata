import torch

from kogata.attention import Head

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
