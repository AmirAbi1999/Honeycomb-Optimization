"""Band gaps of a solved dispersion relation, and the targets measured on them.

Frequencies arrive as (n_wave_numbers, n_bands). Gap i lies between bands i and
i + 1: its edges are the highest band i and the lowest band i + 1 reach over
every wave number. The single-objective run measures each gap's relative width;
the front measures its width and midgap, and reads them as front points.

This module contains:
    - BandGaps
    - measure_band_gaps
    - relative_widths
    - width_midgap_pairs
    - front_points
    - front_gaps
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


@dataclass(frozen=True)
class BandGaps:
    """The gap between every pair of adjacent bands, lowest pair first.

    Attributes
    ----------
    lower_edges, upper_edges : numpy.ndarray
        Highest frequency of the lower band and lowest of the upper band over
        every wave number, in hertz, each of length n_bands - 1.
    """

    lower_edges: np.ndarray
    upper_edges: np.ndarray

    @property
    def widths(self) -> np.ndarray:
        """Return each gap's width in hertz, negative where its bands overlap."""
        return self.upper_edges - self.lower_edges

    @property
    def midgaps(self) -> np.ndarray:
        """Return each gap's midgap frequency, in hertz."""
        return (self.upper_edges + self.lower_edges) / 2

    @property
    def relative_widths(self) -> np.ndarray:
        """Return each gap's width over its midgap frequency, dimensionless.

        Raises
        ------
        ValueError
            If a gap has both of its edges at zero, which leaves its relative
            width undefined.
        """
        midgaps = self.midgaps

        # Only the ratio needs a non-zero midgap, so the check lives here
        # rather than blocking the widths a multi-objective run reads
        if np.any(midgaps == 0):
            raise ValueError(
                "A relative width needs a non-zero midgap; gaps "
                f"{np.flatnonzero(midgaps == 0).tolist()} have both edges at zero."
            )

        return self.widths / midgaps


def measure_band_gaps(frequencies: np.ndarray) -> BandGaps:
    """Find the edges of every gap between adjacent bands.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies of shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    BandGaps
        Edges of the n_bands - 1 gaps.
    """
    # An eigensolver returns the modes at one wave number in whatever order it
    # converged them, and a gap only means anything once they ascend
    frequencies = np.sort(frequencies, axis=1)

    # A complete gap has to survive every wave number, so each edge is bounded
    # by the worst one
    return BandGaps(
        lower_edges=frequencies[:, :-1].max(axis=0),
        upper_edges=frequencies[:, 1:].min(axis=0),
    )


def relative_widths(frequencies: np.ndarray) -> np.ndarray:
    """Measure the targets of the single-objective run, its inner function h.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies of shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    numpy.ndarray
        Relative width of every gap, of length n_bands - 1.

    Raises
    ------
    ValueError
        If a gap has both of its edges at zero.
    """
    return measure_band_gaps(frequencies).relative_widths


def width_midgap_pairs(frequencies: np.ndarray) -> np.ndarray:
    """Measure the targets of the multi-objective run, its inner function h.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Eigenfrequencies of shape (n_wave_numbers, n_bands), in hertz.

    Returns
    -------
    numpy.ndarray
        Width and midgap of every gap, interleaved as [W1, C1, W2, C2, ...],
        of length 2 * (n_bands - 1).
    """
    gaps = measure_band_gaps(frequencies)

    # Interleaved along the last axis, the layout front_points reads back;
    # stacking along the first would ravel to every width, then every midgap
    return np.stack((gaps.widths, gaps.midgaps), axis=-1).ravel()


def front_points(targets: Tensor, asinh_scale: float | None = None) -> Tensor:
    """Turn interleaved targets into the points of the width-midgap front.

    Parameters
    ----------
    targets : Tensor
        Widths and midgaps interleaved as width_midgap_pairs measures them,
        shape (..., 2 * n_gaps).
    asinh_scale : float, optional
        Hertz at which both axes turn from linear to logarithmic, through
        asinh(value / asinh_scale). None leaves them in hertz.

    Returns
    -------
    Tensor
        One (width, negated midgap) point per gap, shape (..., n_gaps, 2), so
        that both objectives maximize.
    """
    pairs = targets.unflatten(-1, (-1, 2))
    points = torch.stack((pairs[..., 0], -pairs[..., 1]), dim=-1)

    # asinh is odd and increasing, so it keeps every dominance and the sign of
    # the negated midgap, and changes only how much each region weighs
    if asinh_scale is None:
        return points

    return torch.asinh(points / asinh_scale)


def front_gaps(
    targets: np.ndarray,
    reference_point: tuple[float, float],
) -> np.ndarray:
    """List the gaps on the width-midgap front, lowest midgap first.

    Parameters
    ----------
    targets : numpy.ndarray
        Interleaved widths and midgaps of shape (n_designs, 2 * n_gaps), as
        width_midgap_pairs measures them.
    reference_point : tuple of float
        The (width, midgap) in hertz a gap has to beat to count: wider, and
        centred lower.

    Returns
    -------
    numpy.ndarray
        One row per gap on the front, as (design index, gap index, width,
        midgap), both indices counting from zero.
    """
    points = front_points(torch.as_tensor(targets))
    n_gaps = points.shape[-2]
    points = points.reshape(-1, 2)
    width, midgap = reference_point
    reference = torch.tensor([width, -midgap], dtype=points.dtype)

    # A gap short of the reference dominates nothing, and cannot be dominated
    # away by one that also falls short
    on_front = is_non_dominated(points) & (points > reference).all(-1)
    indices = on_front.nonzero().squeeze(-1)
    rows = np.column_stack((
        (indices // n_gaps).numpy(),
        (indices % n_gaps).numpy(),
        points[indices, 0].numpy(),
        -points[indices, 1].numpy(),
    ))

    return rows[np.argsort(rows[:, 3])]
