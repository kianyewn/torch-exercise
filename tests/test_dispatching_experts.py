import pytest
import torch


def create_gates():
    scores = torch.rand(3, 5)
    scores_mean = scores.mean(dim=1, keepdims=True)
    gates = torch.where(scores < scores_mean, 0, scores)
    return gates


@pytest.fixture
def gates():
    return torch.tensor(
        [
            [0.9760, 0.6896, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.7228, 0.0000, 0.8359, 0.0000],
            [0.9151, 0.8389, 0.0000, 0.0000, 0.0000],
        ],
        dtype=torch.float32,
    )


def test_get_nonzero_index(gates):
    index = torch.nonzero(gates)
    assert torch.allclose(
        index, torch.tensor([[0, 0], [0, 1], [1, 1], [1, 3], [2, 0], [2, 1]])
    )


def test_get_index_of_sorted_experts(gates):
    sorted_expert_index, _ = torch.nonzero(gates).sort(dim=0)
    assert torch.allclose(sorted_expert_index[:, 1], torch.tensor([0, 0, 1, 1, 1, 3]))

    # for some reason, possible that people use unbind, or slicing
    # note that unbind does not
    _, expert_index = sorted_expert_index.split(1, dim=1)
    assert torch.allclose(expert_index, torch.tensor([[0], [0], [1], [1], [1], [3]]))


def test_get_index_of_batch_with_experts_1_2_3_so_on(gates):
    # expand_all_the_index_that_uses_expert_0_1_3_so_on
    sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
    expert_index = torch.nonzero(
        gates
    )[
        index_sorted_experts[:, 1]
    ]  # Note that when expanding, you have to use torch.nonzero(), because your sort is based on torch.nonzero()
    assert torch.allclose(
        expert_index, torch.tensor([[0, 0], [2, 0], [0, 1], [1, 1], [2, 1], [1, 3]])
    )


def test_get_index_of_top_k_experts(gates):
    non_zero_gates = torch.nonzero(gates)
    assert torch.allclose(non_zero_gates[:, 1], torch.tensor([[0, 1, 1, 3, 0, 1]]))


def test_get_sorted_index_of_top_k_experts(gates):
    sorted_experts, sorted_experts_index = torch.nonzero(gates).sort(0)
    assert torch.allclose(sorted_experts[:, 1], torch.tensor([0, 0, 1, 1, 1, 3]))
