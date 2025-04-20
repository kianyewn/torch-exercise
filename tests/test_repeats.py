import torch


def test_repeat_interleave():
    length = torch.arange(3)
    repeated = length.repeat_interleave(2)
    assert torch.allclose(repeated, torch.tensor([0, 0, 1, 1, 2, 2]))

    repeated = length.repeat_interleave(3)
    assert torch.allclose(repeated, torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]))

    repeated = torch.arange(10).reshape(2, 5).repeat_interleave(2, dim=0)
    assert torch.allclose(
        repeated,
        torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]
        ),
    )


def test_repeat_vector():
    a = torch.tensor([0, 1, 2])
    repeated = a.repeat(2)
    assert torch.allclose(
        repeated, torch.tensor([0, 1, 2, 0, 1, 2])
    )  # notice the difference between interveave


def test_repeat_2d():
    a = torch.arange(10).reshape(2, 5)
    repeated = a.repeat(2, 2)
    assert not torch.allclose(  # NOTICE IT IS NOT INTERLEAVING THE ROWS AND COLUMNS
        repeated,
        torch.tensor(
            [
                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
            ]
        ),
    )

    assert torch.allclose(
        repeated,
        torch.tensor(
            [
                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
            ]
        ),
    )