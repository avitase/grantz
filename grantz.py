from typing import List, Tuple

import numpy as np


def _out_of_bounds(xy, w, h):
    return (xy[:, 0] < 0) | (xy[:, 0] >= w) | (xy[:, 1] < 0) | (xy[:, 1] >= h)


def _hash(xy, n):
    return xy[:, 0] + xy[:, 1] * n


def validate_update(
    x: List[Tuple[int, int]], dx: List[Tuple[int, int]], *, world_size: Tuple[int, int]
) -> List[bool]:
    """
    Validates updates x + dx

    An update is valid iff:
      (1) the updated position is within the bounds of the world
      (2) updated positions are unique
      (3) it is not a swapping move (A goes to B, and B goes to A)

    :param x: Current position as a list of integer pairs
    :param dx: Update step as a list of integer pairs
    :param world_size: Length and height of the world
    :return: Mask of valid updates
    """
    n = len(x)
    assert n == len(dx)

    w, h = world_size
    assert w > 0 and h > 0

    x = np.array(x).astype(int)
    dx = np.array(dx).astype(int)

    # sanity check: are positions out of bounds?
    assert not np.any(_out_of_bounds(x, w, h))

    y = x + dx
    hash_x = _hash(x, h)

    # sanity check: are positions unique?
    assert np.all(np.unique(hash_x, return_counts=True)[1] == 1)

    # condition (1)
    invalid = _out_of_bounds(y, w, h)
    y[invalid] = x[invalid]
    hash_y = _hash(y, h)

    # condition (2)
    v, k = np.unique(hash_y, return_counts=True)
    invalid |= np.isin(hash_y, v[k > 1])
    hash_y[invalid] = _hash(y[invalid], h)

    # condition (3)
    src = np.repeat(hash_x[:, np.newaxis], n, axis=1)
    dst = np.repeat(hash_y[:, np.newaxis], n, axis=1)
    t = dst == src.T
    invalid |= np.any(t & t.T, axis=0)

    return np.invert(invalid).tolist()
