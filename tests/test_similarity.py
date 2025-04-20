import torch


def test_similarity_wrong_before_norm():
    a1 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float32)
    similarity = torch.matmul(a1, a1.T)
    assert torch.allclose(
        similarity,
        torch.tensor(
            [[1.0, 3.0, 5.0], [3.0, 13.0, 23.0], [5.0, 23.0, 41.0]], dtype=torch.float32
        ),
    )


def test_similarity_correct_after_norm():
    a1 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float32)
    a1_norm = a1 / torch.norm(a1, p=2, dim=1, keepdim=True)
    similarity = torch.matmul(a1_norm, a1_norm.T)
    assert torch.allclose(
        similarity,

        torch.tensor(
            [
                [1.0000, 0.8321, 0.7809],
                [0.8321, 1.0000, 0.9962],
                [0.7809, 0.9962, 1.0000],
            ],
            dtype=torch.float32,
        ),
        atol= 1e-04
    )


def test_similarity_the_same_after_columwise_norm_or_layerwise_norm():
    """Show that dotproduct similarity using normalized features will not return the correct similarity between vectors

    Columnwise normalization and layerwise normalization will give the same similarity FOR ALL BATCHES. WHICH IS WRONG
    """
    a1 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float32)
    a1_mean = a1.mean(dim=1, keepdim=True)  # This is layerwise norm
    a1_std = a1.std(dim=1, keepdim=True)
    a1_norm = (a1 - a1_mean) / a1_std
    similarity = torch.matmul(a1_norm, a1_norm.T)
    assert torch.allclose(
        similarity, torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    )
    # torch.argmax(torch.matmul(a1, a1.T), dim=1)


def test_similarity_incorrect_after_rowwise_or_batchnorm():
    """Show that similarity is incorrect if we do rowwise normalization or batch normalization"""
    a1 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float32)
    a1_mean = a1.mean(dim=0)  # this is batch norm
    a1_std = a1.std(dim=0)
    a1_norm = (a1 - a1_mean) / a1_std
    similarity = torch.matmul(a1_norm, a1_norm.T)
    assert torch.allclose(
        similarity,
        torch.tensor(
            [[2.0, 0.0, -2.0], [0.0, 0.0, 0.0], [-2.0, 0.0, 2.0]], dtype=torch.float32
        ),
    )
