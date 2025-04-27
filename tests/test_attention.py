import numpy as np
import torch
import torch.nn.functional as F


def test_local_attention_computation():
    """Applying Windowing to get local attention: Notice that we do not do query padding
    1. Query at time t only looks at its window of keys/values nearby.
    2. Note: q usually stays (B, N, D), but if needed for broadcasting, we can treat it as (B, N, 1, D) too.
    3. Softmax is based on the window_size of similarity matrix, not the timestep (of query).
    4. Output shape is still the same
    """
    query = torch.arange(1 * 3 * 4).reshape(1, 3, 4).float()
    key = torch.arange(1 * 3 * 4).reshape(1, 3, 4).float()
    value = torch.arange(1 * 3 * 4).reshape(1, 3, 4).float()

    b, t, d = query.shape
    look_backward = 1
    look_forward = 0
    pad_value = -1
    window_size = look_backward + look_forward + 1
    assert window_size == 2

    ## Applying Windowing to get local attention: Notice that we do not do query padding
    ### Query at time t only looks at its window of keys/values nearby.
    ### Note: q usually stays (B, N, D), but if needed for broadcasting, we can treat it as (B, N, 1, D) too.

    # key pad
    key_pad = torch.nn.functional.pad(
        key, (0, 0, look_backward, look_forward), value=pad_value
    )
    key_w = key_pad.unfold(dimension=1, size=window_size, step=1).transpose(
        -1, -2
    )  # (B, T, window_size, d)

    # Value pad
    value_pad = torch.nn.functional.pad(
        value, (0, 0, look_backward, look_forward), value=pad_value
    )
    value_w = value_pad.unfold(dimension=1, size=window_size, step=1).transpose(
        -1, -2
    )  # (B, T, window_size, d)

    # Calculate similarity matrix of each timestep to each window_size (as opposed to Timestep)
    simi = query.unsqueeze(-2) @ key_w.transpose(
        -1, -2
    )  # (B, T,1, D) @ (B, T, D, window_size) -> (B, T, 1, window_size)
    simi = torch.nn.functional.softmax(
        simi.float(), dim=-1
    )  # (soft-max over window_size, because it is the timestamp)
    attention = (
        simi @ value_w
    )  # (B, T, 1, window_size), (B,T, window_size, d)) -> (B, T, 1, d)
    out = attention.squeeze(-2)  # (B, T, 1, d) -> (B, T, d)
    assert out.shape == (b, t, d)
    out.shape


def test_simple_causal_mask():
    # x = torch.randn(5,5)
    x = torch.tensor(
        [
            [0.9796, -1.9249, -0.0535, 1.3984, 0.3258],
            [0.1398, 0.2296, 1.7846, -1.3503, 2.1636],
            [1.3536, 1.2026, -1.1259, -0.0070, -1.5591],
            [-0.7320, 0.9097, 0.2079, 0.1151, 1.2707],
            [0.4202, -0.7859, -0.5197, 0.0703, -0.4581],
        ]
    )

    actual = x.masked_fill_(torch.tril(torch.ones(5, 5)) == 0, -1e9).softmax(
        dim=-1
    )  # very large negative value and then softmax
    expected = torch.tensor(
        [
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.4776, 0.5224, 0.0000, 0.0000, 0.0000],
            [0.5145, 0.4424, 0.0431, 0.0000, 0.0000],
            [0.0904, 0.4670, 0.2315, 0.2110, 0.0000],
            [0.3558, 0.1065, 0.1390, 0.2508, 0.1478],
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-4)


def test_attention_mask_with_causal_mask():
    causal_mask = torch.tril(torch.ones(5, 5))
    assert torch.allclose(
        causal_mask,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    # attention_mask = torch.randint(0,2, (5,5)).sort(dim=-1, descending=True)[0]
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )

    final_mask = (
        causal_mask.bool() & attention_mask.bool()
    ).float()  # Notice how only positions that are unmaksed are masked

    assert torch.allclose(
        final_mask,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
            ]
        ),
    )


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = padded_x.unfold(1, forward + backward + 1, 1)
    return tensors.movedim(-1, dim).flatten(dim, dim + 1)


def test_local_attention_with_mask():
    # look around mask too
    # attention_mask = torch.randint(0,2, (2,5)).sort(dim=-1, descending=True)[0] # (B, T)
    attention_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    mask_look = look_around(
        attention_mask.unsqueeze(-1).float(), backward=1, forward=0, pad_value=0
    )  # (B, T, 2)
    # mask_look = mask_look.transpose(-1, -2)  # (B, T, 2)
    assert torch.allclose(
        mask_look,
        torch.tensor(
            [
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )

    attn_logits = torch.randn(
        2, 5, 1, 2
    )  # (B, T, 1, window_size) # similarity between window and timestep
    attn_logits = attn_logits.masked_fill(
        ~mask_look.unsqueeze(-2).bool(), -np.float32("inf")
    ).softmax(dim=-1)
    # mask_look is a mask of the window, so if all in a window is masked, then all in the window is masked. (e.g lookback=1,  batch1_mask = [1, 1, 1, 0, 0])
    attn_logits = torch.where(
        attn_logits.isnan(), torch.tensor(0), attn_logits
    )  # it is possible that all in a window, all is masked because of the attention_mask

    key = torch.randn(2, 5, 2, 3)  # (B, T, window_size, d)
    out = attn_logits @ key  # (B, T, 1, window_size), (B, T, window_size, d)  ->
    out = out.squeeze()
    assert torch.allclose(
        out[1][-2],  # (D,)
        key[1][-2][0],  # (D,) # key[1][-2][1] is masked
    )
