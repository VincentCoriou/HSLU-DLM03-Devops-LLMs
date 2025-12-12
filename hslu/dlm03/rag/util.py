"""Utility functions for working with numpy arrays."""

from collections.abc import Sequence
from typing import ParamSpec, TypeVar

import numpy as np

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def expand_match_dims(*arrays: np.ndarray, sizes: Sequence[int] | None = None) -> list[np.ndarray]:
    """Expands the dimensions of numpy arrays to make them compatible for broadcasting.

    This function takes a variable number of numpy arrays and adds new axes
    to each array so that they can be broadcast together. The final shape of
    each array will be determined by the `sizes` argument, which specifies
    the number of dimensions for each input array.

    Args:
        *arrays: A variable number of numpy arrays.
        sizes: A list of integers specifying the number of dimensions for each
            input array. If None, the number of dimensions will be inferred
            from each array's `ndim` attribute.

    Returns:
        A list of new numpy arrays with expanded dimensions.

    Raises:
        ValueError: If the length of `sizes` does not match the number of
            input arrays.
    """
    if sizes is None:
        sizes = [array.ndim for array in arrays]
    elif len(sizes) != len(arrays):
        error_message = f"Expected {len(arrays)} sizes, but got {len(sizes)}."
        raise ValueError(error_message)
    keep = (slice(None),)
    new = (None,)
    new_arrays = []
    for i, array in enumerate(arrays):
        prefix_dims = sum(sizes[:i])
        suffix_dims = sum(sizes[i + 1:])
        keep_dims = sizes[i]
        new_arrays.append(array[*(new * prefix_dims), *(keep * keep_dims), *(new * suffix_dims), ...])
    return new_arrays


def expand_match_broadcast(*arrays: np.ndarray, sizes: Sequence[int] | None = None) -> list[np.ndarray]:
    """Expands and broadcasts numpy arrays to a common shape.

    This function first expands the dimensions of the input arrays using
    `expand_match_dims` and then broadcasts each array to a common shape.
    The final shape is determined by the shapes of all input arrays.

    Args:
        *arrays: A variable number of numpy arrays.
        sizes: A list of integers specifying the number of dimensions for each
            input array. If None, the number of dimensions will be inferred
            from each array's `ndim` attribute.

    Returns:
        A list of new numpy arrays that have been expanded and broadcast
        to a common shape.
    """
    if sizes is None:
        sizes = [array.ndim for array in arrays]
    base_shape = []
    for array, size in zip(arrays, sizes, strict=True):
        base_shape.extend(array.shape[:size])
    arrays = expand_match_dims(*arrays, sizes=sizes)
    new_arrays = []
    for i, array in enumerate(arrays):
        new_array = array
        for j in range(sum(sizes[:i])):
            new_array = new_array.repeat(base_shape[j], axis=j)
        offset = sum(sizes[: i + 1])
        for j in range(offset, offset + sum(sizes[i + 1:])):
            new_array = new_array.repeat(base_shape[j], axis=j)
        new_arrays.append(new_array)
    return new_arrays
