"""The outer functions g of BO-CF, each with the acquisition it implies.

An objective scores observed targets of shape (n_designs, n_targets), and
builds the acquisition that chases a better score from the fitted surrogates.
Both apply g to observed data or to posterior samples, never to a posterior
mean, which would quietly discard the composite structure.

This module contains:
    - MaximumObjective
    - HypervolumeObjective
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.safe_math import fatminimum, log_fatplus, logmeanexp
from torch import Tensor


@dataclass(frozen=True)
class MaximumObjective:
    """The largest target any design reaches, chased by expected improvement.

    Attributes
    ----------
    score_name : str
        What the score is called in the log.
    mc_samples : int
        Quasi-random posterior samples the expectation is taken over.
    """

    score_name: str = "largest target"
    mc_samples: int = 256

    @staticmethod
    def score(targets: np.ndarray) -> float:
        """Return the largest of the observed targets."""
        return float(targets.max())

    def acquisition(
        self,
        model: SingleTaskGP,
        targets: Tensor,
        seed: int,
    ) -> qLogExpectedImprovement:
        """Build the expected improvement of a design's largest target.

        Parameters
        ----------
        model : botorch.models.SingleTaskGP
            Surrogates fitted to the targets, one output per column.
        targets : Tensor
            Observed targets of shape (n_designs, n_targets).
        seed : int
            Fixes the posterior draw, so one optimization sees one surface.

        Returns
        -------
        botorch.acquisition.logei.qLogExpectedImprovement
            Acquisition to maximize at q = 1, which accepts pending points.
        """
        return qLogExpectedImprovement(
            model=model,
            best_f=targets.max(),
            sampler=SobolQMCNormalSampler(torch.Size([self.mc_samples]), seed=seed),
            objective=GenericMCObjective(lambda samples, X=None: samples.amax(-1)),
        )


@dataclass(frozen=True)
class HypervolumeObjective:
    """The area a two-objective front dominates, chased by qLogNEHVI.

    Each design's targets occupy one or more points of the front, both
    coordinates maximized, and only what lies beyond the reference point
    counts.

    Attributes
    ----------
    front_points : callable
        Maps targets of shape (..., n_targets) to the points they occupy,
        shape (..., n_points, 2).
    reference_point : Tensor
        The two coordinates a point has to beat, in front_points' units.
    score_name : str
        What the score is called in the log.
    mc_samples : int
        Quasi-random posterior samples the expectation is taken over.
    """

    front_points: Callable[[Tensor], Tensor]
    reference_point: Tensor
    score_name: str = "front hypervolume"
    mc_samples: int = 128

    def score(self, targets: np.ndarray) -> float:
        """Return the area the observed targets' points dominate."""
        reference = self.reference_point.numpy()

        # A point short of the reference on either side dominates nothing, and
        # clamping it there zeroes its share without a separate filter
        points = self.front_points(torch.as_tensor(targets)).reshape(-1, 2).numpy()
        points = np.maximum(points, reference)

        # Largest first coordinate first, so each point adds only the height
        # it reaches above every point before it, and the staircase is exact
        ordered = points[np.argsort(-points[:, 0], kind="stable")]
        heights = np.maximum.accumulate(ordered[:, 1])
        steps = np.diff(heights, prepend=reference[1])

        return float(np.sum((ordered[:, 0] - reference[0]) * steps))

    def acquisition(
        self,
        model: SingleTaskGP,
        targets: Tensor,
        seed: int,
    ) -> qLogNoisyExpectedHypervolumeImprovement:
        """Build the expected hypervolume a design's points add to the front.

        Parameters
        ----------
        model, targets, seed
            As MaximumObjective.acquisition takes them.

        Returns
        -------
        botorch.acquisition.multi_objective.qLogNoisyExpectedHypervolumeImprovement
            Acquisition to maximize at q = 1, which accepts pending points.
        """
        objective = _FrontPoints(self.front_points)

        return _SweptHypervolumeImprovement(
            observed_front=objective(targets),
            n_parameters=model.train_inputs[0].shape[-1],
            model=model,
            ref_point=self.reference_point,
            sampler=SobolQMCNormalSampler(torch.Size([self.mc_samples]), seed=seed),
            objective=objective,
        )


class _FrontPoints(MCMultiOutputObjective):
    """Fan one design out into the front points its targets occupy.

    Parameters
    ----------
    front_points : callable
        As HypervolumeObjective holds it.
    """

    def __init__(self, front_points: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.front_points = front_points

        # One design leaves several points behind, so the output carries more
        # q entries than X; BoTorch stands its shape check down the same way
        # for its own set-valued risk measures
        self._verify_output_shape = False

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        """Return the (..., q * n_points, 2) front points of every sample.

        Parameters
        ----------
        samples : Tensor
            Posterior draws of shape (..., q, n_targets).
        X : Tensor, optional
            Unused, and only in the signature BoTorch calls through.
        """
        # On the samples, never on a mean, so any rescaling stays inside g
        return self.front_points(samples).flatten(-3, -2)


class _SweptHypervolumeImprovement(qLogNoisyExpectedHypervolumeImprovement):
    """qLogNEHVI against the observed front, with each cell's gain swept.

    Solved designs enter at their observed values, not as posterior samples
    drawn with every candidate, whose cost grows with the square of the
    designs solved. Pending designs are still sampled once and folded into
    every sample's front, so each design in a batch sees the ones before it.
    Within a cell, the boxes new points cover form a staircase one sort sizes
    exactly, where BoTorch enumerates 2 ** n_points - 1 subsets.

    Parameters
    ----------
    observed_front : Tensor
        Points the solved designs' targets occupy, shape (n_points, 2), in the
        objective's units.
    n_parameters : int
        Width of a design, which the empty baseline has to carry.
    **keywords
        Passed through to qLogNoisyExpectedHypervolumeImprovement.
    """

    # torch types every module attribute as a tensor or a module, which
    # hides what these three are from a type checker
    objective: MCMultiOutputObjective
    cell_lower_bounds: Tensor
    cell_upper_bounds: Tensor

    def __init__(
        self,
        observed_front: Tensor,
        n_parameters: int,
        **keywords: Any,
    ) -> None:
        # BoTorch partitions the front from inside its own __init__, so the
        # observed points have to be in place before it runs
        self.observed_front = observed_front
        super().__init__(
            X_baseline=observed_front.new_empty(0, n_parameters), **keywords,
        )

    def _set_cell_bounds(self, num_new_points: int) -> None:
        """Partition the observed front, with each sample's pending points.

        Parameters
        ----------
        num_new_points : int
            Pending designs added since the last partition, as BoTorch counts
            them.
        """
        super()._set_cell_bounds(num_new_points)

        # The baseline holds only pending designs, so the front BoTorch just
        # partitioned is theirs alone, one per sample
        pending = self.partitioning.pareto_Y
        observed = self.observed_front.expand(len(pending), -1, -1)
        self.partitioning = FastNondominatedPartitioning(
            ref_point=self.ref_point,
            Y=torch.cat((observed, pending), dim=-2),
        )
        bounds = self.partitioning.get_hypercell_bounds().view(
            2, *self._batch_sample_shape, -1, self.ref_point.shape[-1],
        )
        self.register_buffer("cell_lower_bounds", bounds[0])
        self.register_buffer("cell_upper_bounds", bounds[1])

    def _compute_log_qehvi(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        """Average the hypervolume each posterior sample gains, in log space.

        Parameters
        ----------
        samples : Tensor
            Posterior draws at the candidates, shape
            (mc_samples, ..., q, n_targets).
        X : Tensor, optional
            Candidates the draws were taken at, passed to the objective.

        Returns
        -------
        Tensor
            Log expected hypervolume improvement of each candidate batch.
        """
        points = self.objective(samples, X=X)

        # One box decomposition per posterior sample, lined up against the
        # sample axis and broadcast over the candidate batch; the outer boxes
        # run to infinity, and are capped where BoTorch caps them in double
        shape = (len(self.cell_lower_bounds), *[1] * (points.ndim - 3), -1, 1, 2)
        lower = self.cell_lower_bounds.view(shape)
        upper = self.cell_upper_bounds.clamp_max(1e10).view(shape)

        # The same smoothed clamp qLogEHVI puts on every point, so a candidate
        # far behind the front still hands the optimizer a gradient
        sides = fatminimum(
            log_fatplus(points.unsqueeze(-3) - lower, tau=self.tau_relu),
            (upper - lower).log(),
            tau=self.tau_max,
        ).exp()

        # Widest box first, so each one adds only the height it reaches above
        # every wider box before it
        order = sides[..., 0].argsort(dim=-1, descending=True)
        ordered = sides.gather(-2, order.unsqueeze(-1).expand_as(sides))
        heights = ordered[..., 1].cummax(dim=-1).values
        steps = heights.diff(dim=-1, prepend=torch.zeros_like(heights[..., :1]))
        areas = (ordered[..., 0] * steps).sum(-1)

        return logmeanexp(areas.sum(-1).log(), dim=0)
