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


def bucket_by_index():
    x = torch.randint(10, (50,))
    # x = (torch.tensor([4, 7, 4, 5, 8, 9, 0, 0, 8, 3, 1, 9, 3, 1, 9, 8, 4, 6, 6, 5, 0, 1, 4, 8, 1, 4, 1, 5, 8, 6, 0, 3, 0, 4, 1, 0, 3, 7, 9, 6, 4, 1, 3, 3, 7, 0, 2, 2, 3, 0]))

    counts = torch.bincount(x)
    sorted_x = torch.sort(x)[0]
    bins = sorted_x.split(split_size=counts.tolist(), dim=0)
    expected = (
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
        torch.tensor([1, 1, 1, 1, 1, 1, 1]),
        torch.tensor([2, 2]),
        torch.tensor([3, 3, 3, 3, 3, 3, 3]),
        torch.tensor([4, 4, 4, 4, 4, 4, 4]),
        torch.tensor([5, 5, 5]),
        torch.tensor([6, 6, 6, 6]),
        torch.tensor([7, 7, 7]),
        torch.tensor([8, 8, 8, 8, 8]),
        torch.tensor([9, 9, 9, 9]),
    )

    for actual, expected in zip(bins, expected):
        assert torch.allclose(actual, expected)


def cumulative_sum_into_bins():
    preds = torch.tensor([0, 1, 3, 3, 4, 4])
    target = torch.tensor([3, 2, 3, 0, 1, 2])
    out, inv, counts = torch.unique(-preds, return_inverse=True, return_counts=True)
    ranked = torch.zeros_like(counts, dtype=torch.float32)
    ranked.scatter_add_(0, inv, target.to(dtype=ranked.dtype))

    # expected
    r0 = target[-1] + target[-2]
    r1 = target[-3] + target[-4]
    r3 = target[1]
    r4 = target[0]

    assert torch.torch.allclose(
        ranked, torch.tensor([r0, r1, r3, r4], dtype=torch.float32)
    )


def test_get_non_zero_values():
    gates = torch.tensor(
        [
            [0.9760, 0.6896, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.7228, 0.0000, 0.8359, 0.0000],
            [0.9151, 0.8389, 0.0000, 0.0000, 0.0000],
        ]
    )
    non_zero_indices = torch.nonzero(gates)
    assert torch.allclose(
        gates[non_zero_indices.unbind(dim=-1)],
        torch.tensor([0.9760, 0.6896, 0.7228, 0.8359, 0.9151, 0.8389]),
    )


def test_bincounts_ignore_sequence():
    # x = torch.tensor([0,0,0,1,1,2,2,5,12]) + 5
    x = torch.tensor([5, 5, 5, 6, 6, 7, 7, 10, 17])
    x2 = x - x.min()
    x2_unique = torch.unique(x2)
    counts = torch.bincount(x2)
    assert torch.allclose(
        counts, torch.tensor([3, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    )  # notice bincounts iaccording to in torch.arange()
    assert torch.all(
        counts[x2_unique] == torch.tensor([3, 2, 2, 1, 1])
    )  # notice that these are the counts that we care about only


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


def test_sort_all_tensors_according_to_index():
    pass


def test_sort_non_zero_value_column_indexes():
    # scores = torch.rand(3,5)
    # scores = torch.where(scores > scores.mean(dim=1, keepdims=True), scores, 0)
    scores = torch.tensor(
        [
            [0.0000, 0.0000, 0.8457, 0.0000, 0.8547],
            [0.0000, 0.8231, 0.9371, 0.0000, 0.0000],
            [0.8564, 0.5664, 0.0000, 0.0000, 0.5510],
        ]
    )
    non_zero_indices = torch.nonzero(scores)
    sorted_non_zeros, sorted_non_zero_indices = torch.sort(non_zero_indices, dim=0)
    reordered_non_zero_according_to_non_zero_values = non_zero_indices[
        sorted_non_zero_indices[:, 1],
    ]
    assert torch.allclose(
        reordered_non_zero_according_to_non_zero_values,
        torch.tensor([[2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 4], [2, 4]]),
    )
