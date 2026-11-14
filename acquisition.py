"""Batches of feasible designs, proposed by maximizing an acquisition.

Every design is a q = 1 maximization by SLSQP over the space's bounds and its
signed constraint margins, started from the most promising feasible Sobol
draws and conditioned on the designs already proposed.

This module contains:
    - SearchSpace
    - propose_batch
"""

from __future__ import annotations

from typing import Protocol

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor


class SearchSpace(Protocol):
    """What a BO-CF run needs of the space it searches."""

    @property
    def bounds(self) -> Tensor:
        """Return the searched bounds, shape (2, n_searched), lower row first."""

    def sample(
        self,
        n_draws: int,
        *,
        seed: int | None = None,
        clearance: float = 0.0,
    ) -> Tensor:
        """Draw quasi-random designs and keep those clearing every constraint."""

    def margins(self, designs: Tensor) -> Tensor:
        """Return each design's constraint margins, shape (..., n_constraints)."""

    def is_feasible(self, designs: Tensor, *, clearance: float = 0.0) -> Tensor:
        """Return whether each design clears every constraint."""

    def to_cells(self, designs: Tensor) -> Tensor:
        """Turn designs into what the solver is handed, one row per design."""


def propose_batch(
    acquisition: AcquisitionFunction,
    space: SearchSpace,
    batch_size: int,
    *,
    clearance: float = 1e-6,
    seed: int | None = None,
    n_restarts: int = 8,
    n_raw_samples: int = 256,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> Tensor:
    """Propose feasible designs, each conditioned on those before it.

    Parameters
    ----------
    acquisition : botorch.acquisition.AcquisitionFunction
        Acquisition an objective built. The batch is spent on it: it comes
        back conditioned on the designs it proposed.
    space : SearchSpace
        Searched bounds and the constraints every design has to clear.
    batch_size : int
        Designs to propose, normally one per solver worker.
    clearance : float
        Margin the restarts keep in hand and SLSQP aims at, so that its own
        constraint tolerance still lands each answer above the limit itself.
    seed : int, optional
        Fixes the Sobol draws, offset by each design's position so they do
        not all start from the same restarts.
    n_restarts, n_raw_samples : int
        SLSQP restarts per design, and the Sobol draws they are picked from.
    max_iterations, tolerance
        Iteration cap and function tolerance of each restart.

    Returns
    -------
    Tensor
        Designs of shape (batch_size, n_searched).

    Raises
    ------
    ValueError
        If fewer than n_restarts of a draw's candidates are feasible.
    """
    proposals: list[Tensor] = []

    # Greedy rather than a joint q-batch: the acquisition integrates over the
    # pending designs, so each design knows of those before it, and every
    # restart stays a q = 1 problem the constraints can be handed to
    for position in range(batch_size):
        if proposals:
            acquisition.set_X_pending(torch.stack(proposals))

        feasible = space.sample(
            n_raw_samples,
            seed=None if seed is None else seed + position,
            clearance=clearance,
        )
        if len(feasible) < n_restarts:
            raise ValueError(
                f"Only {len(feasible)} of {n_raw_samples} draws clear the "
                f"constraints, short of {n_restarts} restarts; revise the space."
            )

        # SLSQP cannot start outside the feasible set, so the restarts are the
        # most promising feasible draws rather than whatever BoTorch would pick
        with torch.no_grad():
            promise = acquisition(feasible.unsqueeze(1))
        best_promise, best_start = promise.max(0)
        starts = feasible[promise.topk(n_restarts).indices]

        # SLSQP drives an active constraint to zero only to within its own
        # tolerance, so each margin is offset to land the answer inside it
        margins = [
            lambda design, index=index: space.margins(design)[index] - clearance
            for index in range(space.margins(starts[0]).shape[-1])
        ]
        candidate, value = optimize_acqf(
            acquisition,
            bounds=space.bounds,
            q=1,
            num_restarts=n_restarts,
            nonlinear_inequality_constraints=[(margin, True) for margin in margins],
            batch_initial_conditions=starts.unsqueeze(1),
            options={"maxiter": max_iterations, "ftol": tolerance},
        )
        candidate = candidate.squeeze(0)

        # Accepted against the limit itself, since the clearance was already
        # spent on SLSQP's tolerance; a candidate short of the best draw means
        # every restart lost ground, and the draw stands
        if not (bool(space.is_feasible(candidate)) and value > best_promise):
            candidate = feasible[best_start]
        proposals.append(candidate)

    return torch.stack(proposals)
