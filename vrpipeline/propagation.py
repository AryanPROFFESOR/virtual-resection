"""
vrpipeline.propagation
=========================
Extraction of the seizure propagation path from a (binarized) first-order
transfer-entropy graph, following Sun et al.'s definition:

    "step refers to the number of links (edges) connecting an electrode
    to the path of the target electrode... i = seed contact, j = the
    direct neighbor, h = two hop neighbor, k = three hop neighbor"

and the resection ordering rule used throughout their Results and Fig. 2:

    "we cut it off within each layer according to the outdegree (from
    left to right, from top to bottom)"

i.e.: seed node(s) (EZ, clinically marked) -> layer 1 = direct out-
neighbors of the seed -> layer 2 = out-neighbors of layer 1 not already
placed -> layer 3 = out-neighbors of layer 2 not already placed; within
each layer, nodes are ordered by descending outdegree (computed on the
*first-order* directed graph) before being handed to sequential virtual
resection (`resection.sequential_virtual_resection`).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def propagation_layers(binary_directed_adj: np.ndarray, seed_idx: Iterable[int],
                        n_steps: int = 3) -> list:
    """Breadth-first hop-layers outward from `seed_idx` along directed
    edges of `binary_directed_adj` (adj[i, j] == 1 means edge j -> i, i.e.
    j is a source/predecessor of target i -- matches the row=target,
    col=source convention used by `connectivity.transfer_entropy_matrix`).

    Returns
    -------
    list of list[int]: layers[0] is the direct out-neighborhood of the
    seed set (Sun et al.'s "direct neighbor"), layers[1] the two-hop
    neighborhood, etc., up to `n_steps` layers. Nodes already placed in
    an earlier layer (or in the seed set) are excluded from later ones.
    """
    adj = np.asarray(binary_directed_adj)
    n = adj.shape[0]
    seed = set(int(s) for s in seed_idx)
    visited = set(seed)
    layers = []
    frontier = seed
    for _ in range(n_steps):
        nxt = set()
        for node in frontier:
            # node is a source: find targets i such that adj[i, node] != 0
            targets = np.flatnonzero(adj[:, node] != 0)
            for t in targets:
                if t not in visited:
                    nxt.add(int(t))
        nxt -= visited
        if not nxt:
            break
        layers.append(sorted(nxt))
        visited |= nxt
        frontier = nxt
    return layers


def outdegree(binary_directed_adj: np.ndarray) -> np.ndarray:
    """Outdegree of each node on the first-order directed graph (column
    sums, since adj[i, j] = edge j -> i)."""
    return np.asarray(binary_directed_adj).sum(axis=0)


def ordered_removal_sequence(binary_directed_adj: np.ndarray,
                              seed_idx: Iterable[int], n_steps: int = 3
                              ) -> list:
    """Produce the full node-removal order for sequential virtual
    resection: layer-by-layer outward from the seed, and within each
    layer sorted by descending first-order outdegree ("remove
    sequentially from left to right and from top to bottom", Fig. 2d
    caption). Returns a flat list of node indices, seed nodes first."""
    od = outdegree(binary_directed_adj)
    order = list(sorted(seed_idx, key=lambda i: -od[i]))
    for layer in propagation_layers(binary_directed_adj, seed_idx, n_steps):
        order.extend(sorted(layer, key=lambda i: -od[i]))
    return order
