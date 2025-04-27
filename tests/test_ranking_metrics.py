import numpy as np
import torch
from torchmetrics.functional.retrieval import (
    retrieval_normalized_dcg,
    retrieval_precision,
)
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall


def precision_at_k(preds, labels, k):
    _, top_k_indices = torch.topk(preds, k)
    top_k_labels = labels[top_k_indices]
    return top_k_labels.sum() / k


def test_precision_at_k():
    preds = torch.tensor([0.9, 0.8, 0.1])
    labels = torch.tensor([1, 0, 1])
    assert precision_at_k(preds, labels, 3) == 2 / 3
    assert precision_at_k(preds, labels, 2) == 1 / 2

    indexes = torch.tensor([0, 0, 0])
    rp3 = RetrievalPrecision(top_k=3)
    assert rp3(preds, labels, indexes) == 2 / 3
    rp2 = RetrievalPrecision(top_k=2)
    assert rp2(preds, labels, indexes) == 1 / 2


def recall_at_k(preds, labels, k):
    top_k_preds, top_k_indices = torch.topk(preds, k)
    top_k_labels = labels[top_k_indices]
    return top_k_labels.sum() / labels.sum()


def test_recall_at_k():
    preds = torch.tensor([0.9, 0.8, 0.1])
    labels = torch.tensor([1, 0, 1])
    assert recall_at_k(preds, labels, 3) == 1
    assert recall_at_k(preds, labels, 2) == 1 / 2
    rr3 = RetrievalRecall(top_k=3)

    assert rr3(preds, labels, indexes=torch.tensor([0, 0, 0])) == 1
    rr2 = RetrievalRecall(top_k=2)
    assert rr2(preds, labels, indexes=torch.tensor([0, 0, 0])) == 1 / 2


def test_ndcg_raw():
    rel = torch.tensor([3, 2, 3, 0, 1, 2])
    pred = torch.arange(len(rel)).sort(descending=True)[
        0
    ]  # when calculating ndcg, the preds should be sorted in descending order
    cg = torch.sum(rel)
    assert cg == 11

    dcg_denominator = torch.log2(torch.arange(1, len(rel) + 1) + 1)
    assert torch.allclose(
        dcg_denominator,
        torch.tensor([1.0000, 1.5850, 2.0000, 2.3219, 2.5850, 2.8074]),
        atol=1e-4,
    )
    dcg = (rel / dcg_denominator).sum()
    assert torch.allclose(dcg, torch.tensor(6.8611), atol=1e-6)

    # idcg_denominator = torch.log2(torch.arange(1, (rel!=0).sum()+1) +1)
    ideal_rel = torch.sort(rel, descending=True)[0]
    idcg_denominator = torch.log2(torch.arange(1, len(ideal_rel) + 1) + 1)
    idcg = (ideal_rel / idcg_denominator).sum()
    assert torch.allclose(idcg, torch.tensor(7.1410), atol=1e-6)

    ndcg = dcg / idcg
    assert torch.allclose(ndcg, torch.tensor(0.9608), atol=1e-6)

    torch_ndcg = retrieval_normalized_dcg(target=rel, preds=pred.float(), top_k=6)
    assert torch.allclose(torch_ndcg, torch.tensor(0.9608), atol=1e-6)


def test_ndcg_break_ties():
    """Test NDCG with break ties in pred score with `torch.unique()`"""
    # preds = torch.tensor([0, 1, 2, 3, 4, 5])
    preds = torch.tensor([0, 1, 3, 3, 4, 4])
    target = torch.tensor([3, 2, 3, 0, 1, 2])
    out, inv, counts = torch.unique(-preds, return_inverse=True, return_counts=True)
    ranked = torch.zeros_like(counts, dtype=torch.float32)
    ranked.scatter_add_(0, inv, target.to(dtype=ranked.dtype))
    ranked_normalized = ranked / counts
    groups = counts.cumsum(dim=0) - 1
    discount_sums = torch.zeros_like(counts, dtype=torch.float32)

    discount = 1.0 / (
        torch.log2(torch.arange(target.shape[-1], device=target.device) + 2.0)
    )
    discount_cumsum = discount.cumsum(dim=-1)

    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = discount_cumsum[groups].diff()
    gain = (ranked_normalized * discount_sums).sum()

    assert torch.allclose(-preds, torch.tensor([0, -1, -3, -3, -4, -4]))
    assert torch.allclose(
        out, torch.tensor([-4, -3, -1, 0])
    )  # unique values sorted in ascending order by default
    assert torch.allclose(counts, torch.tensor([2, 2, 1, 1]))  # count of unique values
    assert torch.allclose(
        inv, torch.tensor([3, 2, 1, 1, 0, 0])
    )  # representing the indices for where elements in the original input map to in the output
    assert torch.allclose(ranked, torch.tensor([3.0, 3.0, 2.0, 3.0]))
    assert torch.allclose(
        ranked_normalized, torch.tensor([1.5000, 1.5000, 2.0000, 3.0000])
    )
    assert torch.allclose(groups, torch.tensor([1, 3, 4, 5]))
    assert torch.allclose(gain, torch.tensor(5.6847), atol=1e-6)


def ndcg_at_k(rel, pred, k):
    pred, top_pred_indices = torch.topk(pred, k)
    og_rel = rel.detach().clone()
    rel = rel[top_pred_indices]

    # Based on rel of predicted scores
    discount = 1.0 / (torch.log2(torch.arange(len(rel), device=rel.device) + 2.0))
    dcg = (rel * discount).sum()

    # ideal_rel is based on ALL  the relevance
    ideal_rel = og_rel[torch.argsort(og_rel, descending=True)][:k]
    idcg = (ideal_rel * discount).sum()
    ndcg = dcg / idcg
    return ndcg


def test_ndcg_at_k():
    rel = torch.tensor([3, 2, 3, 0, 1, 2])
    pred = torch.arange(len(rel)).float()
    actual_ndcgs = [ndcg_at_k(rel, pred, k=k) for k in range(1, 7)]
    expected_ndcgs = [
        retrieval_normalized_dcg(target=rel, preds=pred, top_k=k) for k in range(1, 7)
    ]
    assert np.allclose(actual_ndcgs, expected_ndcgs)


def test_precision_with_indexes():
    indexes = torch.randint(2, 10, (100,))
    preds = torch.randn(100)
    labels = torch.randint(0, 2, (100,))

    indexes, indices = torch.sort(indexes)
    preds = preds[indices]
    labels = labels[indices]

    counts = torch.bincount(indexes)
    counts = counts[
        counts != 0
    ]  # ignore the 0 counts, works since indexes are already sorted.

    metrics = []
    for mini_preds, mini_labels in zip(
        torch.split(preds, counts.tolist(), dim=0),
        torch.split(labels, counts.tolist(), dim=0),
    ):
        ndcg = retrieval_precision(mini_preds, mini_labels, top_k=5)
        metrics.append(ndcg)

    rp5 = RetrievalPrecision(top_k=5)
    assert torch.allclose(
        torch.mean(torch.stack(metrics)), rp5(preds, labels, indexes=indexes)
    )
