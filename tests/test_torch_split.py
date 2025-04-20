from typing import Tuple

import torch


def test_split_dim1():
    a = torch.arange(10).reshape(5, 2)
    a0: Tuple[torch.Tensor] = a.split(
        1, dim=1
    )  # note that the dimension is not removed, it is just split into 2 tensors
    a1, a2 = a0
    torch.testing.assert_close(a1, torch.tensor([[0], [2], [4], [6], [8]]))
    torch.testing.assert_close(a2, torch.tensor([[1], [3], [5], [7], [9]]))
    assert len(a0) == 2


def test_torch_unbind():
    """Test unbind"""
    a = torch.arange(0, 10).reshape(5, 2)
    unbinded: Tuple[torch.Tensor] = torch.unbind(a, dim=0)  # returns a single tensor
    # note that unbind does not return a list, but a tuple of tensors without the dimension you unbind on
    assert torch.allclose(
        torch.stack(torch.unbind(a, dim=0)),
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),  # note the element
    )
    assert len(unbinded) == 5


def test_tensor_sliciing(a):
    assert torch.allclose(a[[0, 1], [1, 0]], torch.tensor([1, 2]))
    assert torch.allclose(
        a[torch.tensor([0, 1]), torch.tensor([1, 0])], a[[0, 1], [1, 0]]
    )


def test_split_vs_unbind():
    a = torch.arange(10).reshape(5, 2)
    a1, a2 = a.split(1, dim=1)
    torch.testing.assert_close(
        a1, torch.tensor([[0], [2], [4], [6], [8]])
    )  # dim 1 is not removed
    torch.testing.assert_close(a2, torch.tensor([[1], [3], [5], [7], [9]]))

    a3, a4 = a.unbind(dim=1)
    torch.testing.assert_close(a3, torch.tensor([0, 2, 4, 6, 8]))  # dim1 is removed
    torch.testing.assert_close(a4, torch.tensor([1, 3, 5, 7, 9]))
