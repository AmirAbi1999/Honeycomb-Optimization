"""The BO-CF loop, over any search space, solver and objective.

A run solves feasible Sobol designs, then each round fits one surrogate per
target column, proposes batch_size designs and solves them on n_workers
threads, yielding every case solved so far. Each result lands in::

    runs/20261108-143205-front/
    |-- rounds.csv          per round: fit, propose and solve seconds, score
    |-- cases/0001.npz      cell, round, seconds, frequencies, targets or error
    `-- models/round_01.pt  surrogate state_dict fitted in that round

This module contains:
    - RunConfig
    - run
"""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from acquisition import SearchSpace, propose_batch
from objectives import HypervolumeObjective, MaximumObjective
from surrogates import SurrogateConfig, build_surrogates, fit_surrogates

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """How one run is paced, whatever it searches.

    Attributes
    ----------
    surrogate : SurrogateConfig
        Covariance of the surrogates.
    n_initial, n_rounds, batch_size : int
        Sobol designs solved first, rounds after them, designs per round.
    n_workers : int
        Solves in flight at once.
    clearance : float
        Margin every design keeps from the constraint limits.
    seed : int
        Fixes the initial draw; each round offsets it by its own index.
    """

    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    n_initial: int = 40
    n_rounds: int = 8
    batch_size: int = 20
    n_workers: int = 20
    clearance: float = 1e-6
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject a count that would only fail after hours of solves.

        Raises
        ------
        ValueError
            If n_initial is below 2, n_rounds is negative, or batch_size or
            n_workers is not positive.
        """
        for name, minimum in (
            ("n_initial", 2), ("n_rounds", 0), ("batch_size", 1), ("n_workers", 1),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(
                    f"{name} must be at least {minimum}; received "
                    f"{getattr(self, name)}."
                )


def run(
    solve: Callable[[Tensor], np.ndarray],
    measure_targets: Callable[[np.ndarray], np.ndarray],
    objective: MaximumObjective | HypervolumeObjective,
    space: SearchSpace,
    config: RunConfig,
    directory: Path,
) -> Iterator[tuple[list[int], Tensor, Tensor]]:
    """Drive one BO-CF run to its budget, yielding its cases round by round.

    Parameters
    ----------
    solve : callable
        Maps one cell, as space.to_cells builds it, to the solver's output.
    measure_targets : callable
        The inner function h, mapping one solver output to a row of targets.
    objective : MaximumObjective or HypervolumeObjective
        The outer function g, which scores the targets and builds the
        acquisition every round maximizes.
    space : SearchSpace
        What the run searches.
    config : RunConfig
        How the run is paced.
    directory : pathlib.Path
        Folder the run saves into, created if absent.

    Yields
    ------
    list of int, Tensor, Tensor
        Case number, design and targets of every case solved so far, after
        the initial cases and after every round.

    Raises
    ------
    ValueError
        If the space does not yield n_initial feasible designs.
    RuntimeError
        If a round solves none of its designs, or the initial cases leave
        fewer than two to fit.
    """
    for folder in ("cases", "models"):
        (directory / folder).mkdir(parents=True, exist_ok=True)

    log.info("Run folder %s", directory)
    log.info(
        "%d initial cases, then %d rounds of %d on %d workers, seed %d",
        config.n_initial, config.n_rounds, config.batch_size, config.n_workers,
        config.seed,
    )

    # Oversample rather than solve the constraints, since the first feasible
    # draws are the same however many are drawn after them
    n_draws = 64 * config.n_initial
    batch = space.sample(n_draws, seed=config.seed, clearance=config.clearance)
    if len(batch) < config.n_initial:
        raise ValueError(
            f"Only {len(batch)} of {n_draws} draws are feasible, short of the "
            f"{config.n_initial} the run starts from; revise the space."
        )

    batch = batch[:config.n_initial]
    case_ids: list[int] = []
    designs: list[Tensor] = []
    targets: list[np.ndarray] = []
    n_cases = 0

    # Round 0 solves the initial designs; every later round fits and proposes
    # its batch first
    for round_index in range(config.n_rounds + 1):
        label = (
            f"Round {round_index} of {config.n_rounds}"
            if round_index
            else "Initial cases"
        )
        n_fitted = len(designs)
        fit_seconds = propose_seconds = 0.0

        if round_index:
            started = time.perf_counter()
            observed = torch.as_tensor(np.stack(targets))
            model = fit_surrogates(build_surrogates(
                torch.stack(designs), observed, space.bounds, config.surrogate,
            ))
            torch.save(
                model.state_dict(),
                directory / "models" / f"round_{round_index:02d}.pt",
            )
            fit_seconds = time.perf_counter() - started
            log.info(
                "%s: fitted %d surrogates to %d designs in %.1f s",
                label, observed.shape[1], n_fitted, fit_seconds,
            )

            # propose_batch offsets its seed by each design's position, so
            # rounds step batch_size apart to keep their restarts from repeating
            started = time.perf_counter()
            batch = propose_batch(
                objective.acquisition(model, observed, config.seed + round_index),
                space,
                config.batch_size,
                clearance=config.clearance,
                seed=config.seed + round_index * config.batch_size,
            )
            propose_seconds = time.perf_counter() - started
            log.info(
                "%s: proposed %d designs in %.1f s",
                label, len(batch), propose_seconds,
            )

        # A solver call blocks outside the GIL, so threads are enough and the
        # surrogate stays in one process
        started = time.perf_counter()
        cases = range(n_cases + 1, n_cases + 1 + len(batch))
        solve_case = partial(
            _solve_case, solve, measure_targets, objective, directory, round_index,
        )
        with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
            measured = list(pool.map(solve_case, cases, space.to_cells(batch)))
        solve_seconds = time.perf_counter() - started
        n_cases += len(batch)

        for case, design, values in zip(cases, batch, measured):
            if values is not None:
                case_ids.append(case)
                designs.append(design)
                targets.append(values)
        n_solved = len(designs) - n_fitted

        # Two designs are the fewest a surrogate fits to, and a round that
        # solves nothing points at the solver rather than at its designs
        if not n_solved or len(designs) < 2:
            raise RuntimeError(
                f"{label}: {n_solved} of {len(batch)} designs solved, too few to "
                "go on; check the solver before spending more of the budget."
            )

        all_targets = np.stack(targets)
        score = objective.score(all_targets)
        log.info(
            "%s: %d of %d solved in %.1f s, %s %.6g",
            label, n_solved, len(batch), solve_seconds, objective.score_name,
            score,
        )
        _append_row(directory / "rounds.csv", {
            "Round": round_index,
            "Designs fitted": n_fitted,
            "Fit (s)": round(fit_seconds, 1),
            "Propose (s)": round(propose_seconds, 1),
            "Proposed": len(batch),
            "Solved": n_solved,
            "Solve (s)": round(solve_seconds, 1),
            "Score": score,
        })

        yield list(case_ids), torch.stack(designs), torch.as_tensor(all_targets)


def _solve_case(
    solve: Callable[[Tensor], np.ndarray],
    measure_targets: Callable[[np.ndarray], np.ndarray],
    objective: MaximumObjective | HypervolumeObjective,
    directory: Path,
    round_index: int,
    case: int,
    cell: Tensor,
) -> np.ndarray | None:
    """Solve one case, save all it produced, and log how it went.

    Parameters
    ----------
    solve, measure_targets, objective, directory
        As run takes them.
    round_index, case : int
        Round the case belongs to, 0 for the initial designs, and its number.
    cell : Tensor
        One cell, as space.to_cells builds it.

    Returns
    -------
    numpy.ndarray or None
        The case's targets, or None if it did not solve.
    """
    started = time.perf_counter()
    path = directory / "cases" / f"{case:04d}.npz"

    # A cell the solver refuses, or whose output nothing can be measured on,
    # costs one case rather than the whole run
    try:
        output = solve(cell)
        targets = measure_targets(output)
    except Exception as error:
        seconds = time.perf_counter() - started
        np.savez(
            path, cell=cell.numpy(), round=round_index, seconds=seconds,
            error=str(error),
        )
        log.warning("Case %04d failed after %.1f s: %s", case, seconds, error)

        return None

    # Saved under the keys the first runs used, which every script that reads
    # a run's cases goes by
    seconds = time.perf_counter() - started
    np.savez(
        path, cell=cell.numpy(), round=round_index, seconds=seconds,
        frequencies=output, targets=targets,
    )
    log.info(
        "Case %04d solved in %.1f s, %s %.6g",
        case, seconds, objective.score_name, objective.score(targets[None]),
    )

    return targets


def _append_row(path: Path, row: dict[str, float]) -> None:
    """Add one row to a CSV file, writing its header first if the file is new."""
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
