"""Provides functions for calculating evaluation metrics."""

import numpy as np

from hslu.dlm03.rag import util


def recall_at_k(
        *, target_ranks: np.ndarray, k: np.ndarray, mask: np.ndarray | None = None,
) -> np.ndarray:
    """Calculates recall at k.

    Args:
        target_ranks: A numpy array of target ranks.
        k: A numpy array or an integer representing the 'k' in recall@k.
        mask: An optional numpy array to mask values.

    Returns:
        A numpy array with the recall at k values.
    """
    if mask is None:
        mask = np.ones_like(target_ranks)
    if isinstance(k, int):
        k = np.array(k)
    mask, _ = util.expand_match_broadcast(mask, k, sizes=(mask.ndim, k.ndim))
    target_ranks, k = util.expand_match_dims(
        target_ranks, k, sizes=(target_ranks.ndim, k.ndim),
    )
    mask_sum = mask.sum(-2)
    output = np.zeros(mask_sum.shape)
    return np.divide(
        ((target_ranks <= k) * mask).sum(-2),
        mask_sum,
        out=output,
        where=mask_sum > 0,
    )


def precision_at_k(
        *, target_ranks: np.ndarray, k: np.ndarray, mask: np.ndarray | None = None,
) -> np.ndarray:
    """Calculates precision at k.

    Args:
        target_ranks: A numpy array of target ranks.
        k: A numpy array or an integer
        mask: An optional numpy array to mask values.

    Returns:
        A numpy array with the precision at k values.
    """
    if mask is None:
        mask = np.ones_like(target_ranks)
    if isinstance(k, int):
        k = np.array(k)
    mask, _ = util.expand_match_broadcast(mask, k, sizes=(mask.ndim, k.ndim))
    target_ranks, k = util.expand_match_dims(
        target_ranks, k, sizes=(target_ranks.ndim, k.ndim),
    )
    return ((target_ranks <= k) * mask / k).sum(-2)


def mean_rank(*, target_ranks: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Calculates mean rank.

    Args:
        target_ranks: A numpy array of target ranks.
        mask: An optional numpy array to mask values.

    Returns:
        A numpy array with the mean rank values.
    """
    if mask is None:
        mask = np.ones_like(target_ranks)
    output = np.zeros(target_ranks.shape[:-1])
    mask_sum = mask.sum(-1)
    return np.divide(
        (target_ranks * mask).sum(-1),
        mask_sum,
        out=output,
        where=mask_sum > 0,
    )


def rank(targets: np.ndarray, rankings: np.ndarray) -> np.ndarray:
    """Calculates the rank of target items within a ranked list.

    Args:
        targets: A numpy array of target items.
        rankings: A numpy array of ranked items.

    Returns:
        A numpy array with the ranks of the target items.
    """
    indices = np.where(targets[..., None, :] == rankings[..., :, None])
    records = np.rec.fromarrays((*indices[:-2], indices[-1]))
    order = np.argsort(records)
    return indices[-2][order].reshape(targets.shape) + 1
