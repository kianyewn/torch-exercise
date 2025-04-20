import torch


def test_alternate_sign():
    x1 = torch.arange(0, 12).reshape(2, 6)
    x1 = x1.reshape(2, 3, 2)
    a1, a2 = x1.unbind(dim=-1)
    a3 = torch.stack([a1, -a2], dim=-1)
    a4 = a3.reshape(2, -1)
    assert torch.allclose(
        a4, torch.tensor([[0, -1, 2, -3, 4, -5], [6, -7, 8, -9, 10, -11]])
    )


def test_get_values_that_are_not_zero_according_to_column_indices():
    gates = torch.tensor(
        [
            [0.9760, 0.6896, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.7228, 0.0000, 0.8359, 0.0000],
            [0.9151, 0.8389, 0.0000, 0.0000, 0.0000],
        ]
    )
    non_zero_indices = torch.nonzero(gates)
    non_zero_indices_sorted_by_column = non_zero_indices[
        non_zero_indices[:, 1].sort()[1]
    ]
    sorted_values = gates[
        non_zero_indices_sorted_by_column.unbind(dim=-1)
    ]  # indexing needs row and column, thus i unbind it
    # assert torch.allclose(
    #     non_zero_indices_sorted_by_column.unbind(dim=-1),
    #     (
    #         torch.tensor([0, 2, 0, 1, 2, 1]),
    #         torch.tensor([0, 0, 1, 1, 1, 3]),
    #     ),
    # )

    assert torch.allclose(
        sorted_values, torch.tensor([0.9760, 0.9151, 0.6896, 0.7228, 0.8389, 0.8359])
    )

    # alternatively
    non_zero_indices = torch.nonzero(gates)
    sorted_experts, sorted_expert_indices = non_zero_indices.sort(0)
    batch_index = non_zero_indices[sorted_expert_indices[:, 1], 0]
    gates_exp = gates[batch_index]
    gate_values = torch.gather(gates_exp, 1, sorted_experts[:, 1].reshape(-1, 1))
    assert torch.allclose(
        gate_values,
        torch.tensor([[0.9760], [0.9151], [0.6896], [0.7228], [0.8389], [0.8359]]),
    )
