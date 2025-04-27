import torch


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
