import pytest
import torch


@pytest.fixture
def a():
    return torch.arange(0, 10).reshape(5, 2)
