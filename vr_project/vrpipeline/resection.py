"""
vrpipeline.resection
======================
Two virtual-resection schemes, unified into one module because both
ultimately answer the same question -- "what happens to network stability
if we delete this set of nodes?" -- but differ in *which* nodes and *how
many at once*:

  A. En-bloc control centrality (Kini et al.): remove the entire
     clinically-resected electrode set at once and recompute symmetric
     synchronizability s(t). c_res = (s_lesioned - s_base) / s_base.
     Positive = desynchronizing tissue removed; negative = synchronizing
     tissue removed (good-outcome signature per Kini et al. Fig. 5/S2).

  B. Sequential/hierarchical virtual resection (Sun et al.): walk the
     propagation-ordered node sequence (vrpipeline.propagation) one node
     (or one physically-adjacent pair) at a time, recomputing the
     *asymmetric* stability R after each removal, and locate the global
     minimum ("inflection point") -- the smallest node set that drives
     the network to its most stable configuration. Everything after the
     minimum that doesn't improve stability further is excess/avoidable
     resection.

Also implements: electrode-to-resection-zone mapping from real
coordinates + a NIfTI mask (point-in-hull test, exactly reproducing
EpiVR's `util_virtual_resection.get_resected_electrodes`, modernized to
Python 3/scipy), with the dilation/erosion sweep used for the
segmentation-robustness analysis (Kini et al. Supplementary Fig. S3).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import binary_dilation, binary_erosion

from .synchronizability import synchronizability, asymmetric_stability


# ---------------------------------------------------------------------
# Electrode <-> resection-zone mapping (imaging -> node labels)
# ---------------------------------------------------------------------
def _in_hull(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    """Boolean mask: which `points` lie inside the convex hull of
    `hull_points`. Equivalent to EpiVR's `in_hull` used on the
    resection-mask voxel cloud."""
    if len(hull_points) < 4:
        # Degenerate hull (too few voxels); fall back to nearest-voxel
        # membership test.
        from scipy.spatial import cKDTree
        tree = cKDTree(hull_points)
        d, _ = tree.query(points)
        return d < 1.5
    tri = Delaunay(hull_points)
    return tri.find_simplex(points) >= 0


def electrodes_in_resection(mask_vol: np.ndarray, affine: np.ndarray,
                             electrode_coords: dict,
                             dilate_erode_pct: float = 0.0,
                             max_iter: int = 30) -> list:
    """Determine which electrode labels fall inside a binary resection
    mask, in world (scanner) coordinates transformed through `affine`.

    Parameters
    ----------
    mask_vol : ndarray
        Binary resection segmentation volume (see io.load_resection_mask).
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine of the mask NIfTI.
    electrode_coords : dict
        label -> (x, y, z) in the SAME world space as `affine` (i.e.
        electrodes must already be co-registered to the resection-mask
        image -- do that step in ANTs/FreeSurfer/your localization
        pipeline before calling this).
    dilate_erode_pct : float
        Fraction (e.g. 0.20 for +/-20%) to dilate (>0) or erode (<0) the
        mask before testing membership, used for the robustness sweep in
        Kini et al. Supplementary Fig. S3 ("Outcome prediction using
        network measures is robust to error in resection zone mapping").
    max_iter : int
        Safety cap on morphological iterations.

    Returns
    -------
    list[str] of electrode labels inside the (possibly dilated/eroded)
    resection zone.
    """
    inv_affine = np.linalg.inv(affine)
    vol = mask_vol.copy()

    n_iter = 0
    if dilate_erode_pct > 0:
        while n_iter < max_iter:
            hull_vox = np.array(np.where(vol)).T
            if hull_vox.shape[0] == 0:
                break
            labels_out = _electrodes_inside(hull_vox, inv_affine,
                                              electrode_coords)
            frac = len(labels_out) / max(len(electrode_coords), 1)
            if frac >= dilate_erode_pct:
                break
            vol = binary_dilation(vol, iterations=1)
            n_iter += 1
    elif dilate_erode_pct < 0:
        target = max(dilate_erode_pct, -0.99)
        while n_iter < max_iter:
            hull_vox = np.array(np.where(vol)).T
            if hull_vox.shape[0] == 0:
                break
            labels_out = _electrodes_inside(hull_vox, inv_affine,
                                              electrode_coords)
            frac = len(labels_out) / max(len(electrode_coords), 1)
            if frac <= max(0.0, 1.0 + target):
                break
            vol = binary_erosion(vol, iterations=1)
            n_iter += 1

    hull_vox = np.array(np.where(vol)).T
    return _electrodes_inside(hull_vox, inv_affine, electrode_coords)


def _electrodes_inside(hull_vox_ijk: np.ndarray, inv_affine: np.ndarray,
                        electrode_coords: dict) -> list:
    if hull_vox_ijk.shape[0] == 0:
        return []
    labels = list(electrode_coords.keys())
    world_xyz = np.array([electrode_coords[lab] for lab in labels])
    homog = np.hstack([world_xyz, np.ones((len(labels), 1))])
    vox_xyz = (inv_affine @ homog.T).T[:, :3]
    inside = _in_hull(vox_xyz, hull_vox_ijk)
    return [labels[i] for i in np.flatnonzero(inside)]


# ---------------------------------------------------------------------
# A. En-bloc control centrality (Kini et al.)
# ---------------------------------------------------------------------
def lesion_nodes(adj: np.ndarray, node_idx: Iterable[int]) -> np.ndarray:
    """Return the adjacency matrix with `node_idx` rows/cols removed
    (equivalent to Echobase's `Network.Transforms.lesion.node_lesion`)."""
    keep = [i for i in range(adj.shape[0]) if i not in set(node_idx)]
    return adj[np.ix_(keep, keep)]


def region_control_centrality(adj: np.ndarray, node_idx: Iterable[int],
                               base_sync: Optional[float] = None) -> float:
    """c_res = (sync_lesioned - sync_base) / sync_base, en-bloc removal of
    `node_idx` from a symmetric coherence adjacency (Kini et al., verified
    against EpiVR `util_virtual_resection.region_control`)."""
    node_idx = list(node_idx)
    if len(node_idx) == 0 or len(node_idx) >= adj.shape[0] - 1:
        return float("nan")
    if base_sync is None:
        base_sync = synchronizability(adj)
    lesioned = lesion_nodes(adj, node_idx)
    lesioned_sync = synchronizability(lesioned)
    if base_sync == 0 or np.isnan(base_sync):
        return float("nan")
    return (lesioned_sync - base_sync) / base_sync


def nodal_control_centrality(adj: np.ndarray) -> np.ndarray:
    """Per-node control centrality: remove each node individually, one at
    a time, and record delta-synchronizability (Kini et al. node-level
    control centrality used for whole-brain spatial maps, Fig. 6/7)."""
    n = adj.shape[0]
    base_sync = synchronizability(adj)
    out = np.full(n, np.nan)
    for i in range(n):
        out[i] = region_control_centrality(adj, [i], base_sync=base_sync)
    return out


def resection_zone_control_timeseries(adjacency_by_epoch: list,
                                       resected_idx: Iterable[int]
                                       ) -> np.ndarray:
    """c_res(t) across a list of per-epoch symmetric adjacency matrices
    for a fixed resected-node set (Kini et al. Fig. 2 pipeline output)."""
    resected_idx = list(resected_idx)
    return np.array([region_control_centrality(a, resected_idx)
                      for a in adjacency_by_epoch])


# ---------------------------------------------------------------------
# B. Sequential / hierarchical virtual resection (Sun et al.)
# ---------------------------------------------------------------------
def sequential_virtual_resection(hte: np.ndarray, removal_order: list,
                                  smallest_index: int = 1) -> dict:
    """Sequentially delete nodes from `hte` in `removal_order`, recomputing
    asymmetric stability R after each removal (Sun et al. Fig. 2d/4c/5b/
    "control centrality" trend curves).

    Returns
    -------
    dict with:
        'n_removed'      : ndarray 0..len(removal_order)
        'stability'      : ndarray, R after each cumulative removal (R at
                            n_removed=0 is the un-resected baseline)
        'removed_labels' : list of node indices removed at each step
                            (cumulative sets, index-aligned with the above)
        'best_index'     : argmin of 'stability' beyond the baseline (the
                            inflection point / minimum-intervention target)
        'best_removed'   : the cumulative removed-node set at best_index
    """
    hte = np.asarray(hte, dtype=float)
    n = hte.shape[0]
    remaining = list(range(n))
    id_map = {i: i for i in range(n)}  # original idx -> current idx (updates below)

    cur = hte.copy()
    cur_labels = list(range(n))

    stability = [asymmetric_stability(cur, smallest_index)]
    removed_cum = [[]]
    removed_so_far = []

    for node in removal_order:
        if node not in cur_labels:
            continue
        pos = cur_labels.index(node)
        keep = [i for i in range(cur.shape[0]) if i != pos]
        if len(keep) <= smallest_index + 1:
            break
        cur = cur[np.ix_(keep, keep)]
        cur_labels = [cur_labels[i] for i in keep]
        removed_so_far = removed_so_far + [node]
        stability.append(asymmetric_stability(cur, smallest_index))
        removed_cum.append(list(removed_so_far))

    stability = np.array(stability)
    best_index = 0
    if len(stability) > 1:
        tail = stability[1:]
        finite = np.isfinite(tail)
        if finite.any():
            tail_finite = np.where(finite, tail, np.inf)
            best_index = int(np.argmin(tail_finite) + 1)

    return {
        "n_removed": np.arange(len(stability)),
        "stability": stability,
        "removed_labels": removed_cum,
        "best_index": best_index,
        "best_removed": removed_cum[best_index],
    }


def evaluate_alternative_schemes(hte: np.ndarray, schemes: dict,
                                  smallest_index: int = 1) -> dict:
    """Score a set of named candidate resection schemes (node-index lists)
    by their resulting asymmetric stability, exactly the "Evaluation of
    other resection options" comparison in Sun et al. Fig. 2f. Lower R is
    better (network reaches a more stable, less synchronizable state)."""
    scores = {}
    for name, node_idx in schemes.items():
        remaining = [i for i in range(hte.shape[0]) if i not in set(node_idx)]
        if len(remaining) <= smallest_index + 1:
            scores[name] = float("nan")
            continue
        sub = hte[np.ix_(remaining, remaining)]
        scores[name] = asymmetric_stability(sub, smallest_index)
    return scores


def three_stage_comparison(pre_seizure_hte_list: list, seizure_hte_list: list,
                            best_removed: list, smallest_index: int = 1
                            ) -> dict:
    """Pre-seizure vs. seizure vs. virtually-resected-seizure (VR-seizure)
    stability comparison, Sun et al. Fig. 6b/6d."""
    def _score(hte_list, removed):
        vals = []
        for h in hte_list:
            remaining = [i for i in range(h.shape[0]) if i not in set(removed)]
            if len(remaining) <= smallest_index + 1:
                vals.append(np.nan)
                continue
            sub = h[np.ix_(remaining, remaining)]
            vals.append(asymmetric_stability(sub, smallest_index))
        return float(np.nanmedian(vals)) if vals else float("nan")

    pre = np.nanmedian([asymmetric_stability(h, smallest_index)
                         for h in pre_seizure_hte_list])
    sz = np.nanmedian([asymmetric_stability(h, smallest_index)
                        for h in seizure_hte_list])
    vr = _score(seizure_hte_list, best_removed)
    return {"pre_seizure": pre, "seizure": sz, "vr_seizure": vr}
