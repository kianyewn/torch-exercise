import torch


def test_masked_agg():
    # emb = torch.arange(2*3*4).reshape(2,3,4).float() # (B, T, D)
    # emb = torch.where(torch.randn_like(emb)<0, 0, emb)
    emb = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 3.0], [4.0, 0.0, 0.0, 7.0], [8.0, 9.0, 10.0, 0.0]],
            [[12.0, 0.0, 0.0, 0.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
        ]
    )

    sequence_len = 2
    # Mask according to sequence length
    mask = (
        (torch.arange(emb.shape[1]) < sequence_len).reshape(1, -1, 1).float()
    )  # (1,, T, 1)
    masked_emb = emb - (1 - mask) * 1e9 # make masked values very negative, so that when we softmax it will be 0
    assert torch.allclose(
        masked_emb.long(),
        torch.tensor(
            [
                [
                    [0, 0, 2, 3],
                    [4, 0, 0, 7],
                    [-1000000000, -1000000000, -1000000000, -1000000000],
                ],
                [
                    [12, 0, 0, 0],
                    [16, 17, 18, 19],
                    [-1000000000, -1000000000, -1000000000, -1000000000],
                ],
            ]
        ),
    )
    # Agg based on max
    max_agg = masked_emb.max(dim=1)[0]
    assert torch.allclose(
        max_agg, torch.tensor([[4.0, 0.0, 2.0, 7.0], [16.0, 17.0, 18.0, 19.0]])
    )

    # Agg based on sum
    sum_agg = (masked_emb * mask).sum(dim=1) # ignore the masked value
    assert torch.allclose(
        sum_agg, torch.tensor([[4.0, 0.0, 2.0, 10.0], [28.0, 17.0, 18.0, 19.0]])
    )

    mean_agg = (masked_emb * mask).sum(dim=1) / sequence_len
    assert torch.allclose(
        mean_agg,
        torch.tensor(
            [[2.0000, 0.0000, 1.0000, 5.0000], [14.0000, 8.5000, 9.0000, 9.5000]]
        ),
    )
