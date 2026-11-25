"""The honeycomb band-gap optimization, run from the command line.

Joins the honeycomb space, Sara's solver and a band-gap objective for
optimization.run, and writes what only this study keeps in the run's folder::

    runs/20261108-143205-front/
    |-- run.log             the log, line for line
    |-- config.json         run, space, solver and objective settings
    `-- front.csv           gaps on the width-midgap front, multi-objective only

This module contains:
    - main
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, fields
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from band_gaps import front_gaps, front_points, relative_widths, width_midgap_pairs
from design_space import DesignSpace
from dispersion_solver import (
    SolverSettings,
    check_matlab,
    solve_dispersion,
    start_engines,
)
from geometry import HoneycombGeometry
from objectives import HypervolumeObjective, MaximumObjective
from optimization import RunConfig, run

log = logging.getLogger(__name__)


def main() -> None:
    """Check the solver, then run one optimization into a folder of its own."""
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="stop after one solve that proves the setup",
    )
    parser.add_argument(
        "--multi-objective",
        action="store_true",
        help="grow the gap width-midgap front rather than the widest relative gap",
    )
    parser.add_argument(
        "--reference-point",
        type=float,
        nargs=2,
        metavar=("width", "midgap"),
        help=(
            "the gap, in hertz, every gap on the front has to beat: wider, and "
            "centred lower; required with --multi-objective"
        ),
    )
    parser.add_argument(
        "--asinh-scale",
        type=float,
        metavar="hertz",
        help=(
            "where both front axes turn from linear to logarithmic, through "
            "asinh(value / scale); defaults to the reference point's geometric mean"
        ),
    )
    parser.add_argument(
        "--asinh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rescale both front axes through asinh; --no-asinh keeps them in hertz",
    )
    for flag, default, text in (
        ("--initial", defaults.n_initial, "Sobol designs solved before the first fit"),
        ("--rounds", defaults.n_rounds, "fit-propose-solve rounds after that"),
        ("--batch-size", defaults.batch_size, "designs each round proposes"),
        ("--workers", defaults.n_workers, "MATLAB engines solving at once"),
    ):
        parser.add_argument(
            flag, type=int, default=default, metavar="count", help=text,
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        metavar="number",
        help="fixes the initial draw and every round's proposals",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs"),
        metavar="folder",
        help="folder every run gets a subfolder in",
    )
    arguments = parser.parse_args()
    reference_point = arguments.reference_point
    if arguments.multi_objective != (reference_point is not None):
        parser.error("--reference-point goes with --multi-objective, and only there")
    if reference_point is not None and not (
        np.isfinite(reference_point).all() and reference_point[1] > 0.0
    ):
        parser.error("--reference-point takes a finite width and a positive midgap")
    asinh_scale = arguments.asinh_scale
    axis_options_given = asinh_scale is not None or not arguments.asinh
    if axis_options_given and not arguments.multi_objective:
        parser.error(
            "--asinh-scale and --no-asinh go with --multi-objective, and only there",
        )
    if asinh_scale is not None and not arguments.asinh:
        parser.error("--asinh-scale and --no-asinh exclude each other")
    if asinh_scale is not None and not (np.isfinite(asinh_scale) and asinh_scale > 0.0):
        parser.error("--asinh-scale takes a positive number of hertz")

    # The reference point bounds the gaps the front counts, so its geometric
    # mean sets the scale from the problem itself, in any unit
    if asinh_scale is None and arguments.multi_objective and arguments.asinh:
        if reference_point[0] <= 0.0:
            parser.error(
                "--asinh-scale is required when the reference width is not positive",
            )
        asinh_scale = float(np.sqrt(reference_point[0] * reference_point[1]))

    layout = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=layout)

    # GPyTorch and BoTorch report through warnings, which would miss run.log
    logging.captureWarnings(True)

    config = RunConfig(
        n_initial=arguments.initial,
        n_rounds=arguments.rounds,
        batch_size=arguments.batch_size,
        n_workers=arguments.workers,
        seed=arguments.seed,
    )
    space = DesignSpace()
    solver = SolverSettings()

    # A run is hours of solves, so the setup is proved on one cell first
    frequencies = check_matlab(space, solver)
    log.info(
        "MATLAB check passed, relative widths %s",
        np.round(relative_widths(frequencies), 4).tolist(),
    )
    if arguments.check:
        return

    kind = "front" if arguments.multi_objective else "widest-gap"
    directory = arguments.output / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
    directory.mkdir(parents=True)
    file_handler = logging.FileHandler(directory / "run.log")
    file_handler.setFormatter(logging.Formatter(layout))
    logging.getLogger().addHandler(file_handler)

    settings = {"space": asdict(space), "solver": asdict(solver), **asdict(config)}
    if arguments.multi_objective:
        to_points = partial(front_points, asinh_scale=asinh_scale)
        measure_targets = width_midgap_pairs

        # Passed through to_points as a gap of its own, so the reference is
        # rescaled exactly as every gap is
        reference = torch.tensor(reference_point, dtype=torch.double)
        objective = HypervolumeObjective(to_points, to_points(reference)[0])
        settings.update(reference_point=reference_point, asinh_scale=asinh_scale)
        axes = (
            "linear axes in hertz" if asinh_scale is None
            else f"axes turning logarithmic at {asinh_scale:.6g} Hz"
        )
        log.info(
            "The front counts gaps wider than %.6g Hz and centred below %.6g Hz, "
            "on %s",
            *reference_point, axes,
        )
    else:
        measure_targets = relative_widths
        objective = MaximumObjective(score_name="widest relative gap")

    (directory / "config.json").write_text(json.dumps(settings, indent=2))
    log.info(
        "Maximizing the %s over %s from %s to %s, with %s fixed",
        objective.score_name, list(space.searched), space.bounds[0].tolist(),
        space.bounds[1].tolist(), space.fixed,
    )

    started = time.perf_counter()
    with start_engines(config.n_workers) as engines:
        for case_ids, designs, targets in run(
            partial(solve_dispersion, engines, solver),
            measure_targets, objective, space, config, directory,
        ):
            if arguments.multi_objective:
                _write_front(
                    directory / "front.csv", space.to_cells(designs), case_ids,
                    targets.numpy(), reference_point,
                )
    log.info(
        "Run finished in %.1f min, results in %s",
        (time.perf_counter() - started) / 60, directory,
    )


def _write_front(
    path: Path,
    cells: Tensor,
    case_ids: list[int],
    targets: np.ndarray,
    reference_point: tuple[float, float],
) -> None:
    """Rewrite front.csv with every gap on the width-midgap front so far.

    Parameters
    ----------
    path : pathlib.Path
        The front.csv to write.
    cells, case_ids, targets
        Cell, case number and targets of every case solved so far.
    reference_point : tuple of float
        As front_gaps takes it.
    """
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Case", "Lower band", "Upper band", "Width (Hz)", "Midgap (Hz)",
            *(parameter.name for parameter in fields(HoneycombGeometry)),
        ])
        for design, gap, width, midgap in front_gaps(targets, reference_point):
            writer.writerow([
                case_ids[int(design)], int(gap) + 1, int(gap) + 2, width, midgap,
                *cells[int(design)].tolist(),
            ])


if __name__ == "__main__":
    main()
