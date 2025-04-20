"""
torch.nn.functional.pad(input, pad, mode='constant', value=None) → Tensor
pad: Tuple[int, int, ...]
    The size of the padding for each dimension. It is defined as (pad_left, pad_right, pad_top, pad_bottom, ...)
    - Think of it such that the first 2 elements are for the last dimension, the next 2 are for the second last dimension, and so on.
        - I.e. when developing, you should think of padding as padding from larger dimension index to smaller dimension index.
    - If the input is 3D, then the pad is (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def x():
    return torch.arange(12).reshape(2, 3, 2)


def test_pad_3_dims_first_dim(x):
    padded_x = F.pad(x, (0, 0, 0, 0, 1, 1), value=-1)
    assert torch.allclose(
        padded_x,
        torch.tensor(
            [
                [[-1, -1], [-1, -1], [-1, -1]],
                [[0, 1], [2, 3], [4, 5]],
                [[6, 7], [8, 9], [10, 11]],
                [[-1, -1], [-1, -1], [-1, -1]],
            ]
        ),
    )


def test_pad_3_dims_second_dim(x):
    padded_x = F.pad(x, (0, 0, 1, 1, 0, 0), value=-1)
    assert torch.allclose(
        padded_x,
        torch.tensor(
            [
                [[-1, -1], [0, 1], [2, 3], [4, 5], [-1, -1]],
                [[-1, -1], [6, 7], [8, 9], [10, 11], [-1, -1]],
            ]
        ),
    )


def test_pad_3_dims_third_dim(x):
    padded_x = F.pad(x, (1, 1, 0, 0, 0, 0), value=-1)
    assert torch.allclose(
        padded_x,
        torch.tensor(
            [
                [[-1, 0, 1, -1], [-1, 2, 3, -1], [-1, 4, 5, -1]],
                [[-1, 6, 7, -1], [-1, 8, 9, -1], [-1, 10, 11, -1]],
            ]
        ),
    )


def test_pad_3_dims_third_dim_v2(x):
    padded_x = F.pad(x, (1, 1), value=-1)
    assert torch.allclose(
        padded_x,
        torch.tensor(
            [
                [[-1, 0, 1, -1], [-1, 2, 3, -1], [-1, 4, 5, -1]],
                [[-1, 6, 7, -1], [-1, 8, 9, -1], [-1, 10, 11, -1]],
            ]
        ),
    )
