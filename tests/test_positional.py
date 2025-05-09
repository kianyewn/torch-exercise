import math

import numpy as np
import torch


def test_positional_encoding():
    max_seq_len = 100
    dim = 20
    positional_encoding = torch.zeros(max_seq_len, dim)
    positions = torch.arange(max_seq_len)
    half_dim = torch.arange(0, dim, step=2)
    # assert torch.allclose(half_dim, torch.tensor([0, 2, 4, 6, 8]))

    denom = torch.exp(half_dim / dim * -torch.log(torch.tensor(10000.0)))  # (2i,)

    positional_encoding[:, half_dim] = torch.sin(
        positions.unsqueeze(-1) * denom
    )  # every even dimension is a sine
    positional_encoding[:, half_dim + 1] = torch.cos(
        positions.unsqueeze(-1) * denom
    )  # every odd dimension is a cosine

    max_len = max_seq_len
    d_model = dim

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    assert torch.allclose(positional_encoding, pe, atol=1e-5)


def plot(pe):
    """Inspect even dims and odd dimensions along the positions"""
    import matplotlib.pyplot as plt

    pes = pe.numpy()
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(
        np.arange(pes.shape[0]), pe[:, 4:8], label=[f"dim_{i}" for i in [4, 5, 6, 7]]
    )
    return fig
