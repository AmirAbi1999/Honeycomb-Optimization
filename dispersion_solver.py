"""The honeycomb SEM dispersion solver, reached through MATLAB engines.

evaluate_dispersion.m takes its geometry as [hex_angle, cell_width,
face_sheet_thickness, hex_thickness, hex_side_length, hex_side_length], in the
degrees and millimetres a HoneycombGeometry holds, and returns one row of band
frequencies per wave number.

This module contains:
    - SolverSettings
    - start_engines
    - solve_dispersion
    - check_matlab
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import Tensor

from band_gap_objectives import relative_widths, widest_relative_gap
from design_space import DesignSpace
from geometry import HoneycombGeometry
from matlab_caller import MatlabPool

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverSettings:
    """What the SEM dispersion solver is asked for at every design.

    Attributes
    ----------
    delta : float
        Positive resolution factor of the spectral element mesh.
    minimum_polynomial_order : int
        Element order floor, at least 3.
    n_wave_numbers : int
        Wave numbers swept, at least 2.
    n_bands : int
        Bands returned at each wave number, so n_bands - 1 gaps.
    """

    delta: float = 0.27
    minimum_polynomial_order: int = 5
    n_wave_numbers: int = 21
    n_bands: int = 10


def start_engines(n_workers: int) -> MatlabPool:
    """Start MATLAB engines with the solver sources on their path.

    Parameters
    ----------
    n_workers : int
        Engines to start, and so cells that can be solved at once.

    Returns
    -------
    MatlabPool
        Engines whose check confirms every solver function is in reach.
    """
    # Every function evaluate_dispersion.m reaches for that lives beside it
    return MatlabPool(
        Path(__file__).resolve().parent / "src",
        n_workers,
        required_functions=(
            "evaluate_dispersion",
            "generate_geometry",
            "geometry_parameters",
            "get_material_property",
            "element_sampling_3D",
            "element_prepare_3D",
            "Mass_and_Stiffness_Element_3D",
        ),
    )


def solve_dispersion(
    engines: MatlabPool,
    settings: SolverSettings,
    cell: Tensor,
) -> np.ndarray:
    """Solve one cell's dispersion relation, refusing an unusable answer.

    Parameters
    ----------
    engines : MatlabPool
        Engines start_engines started; several threads can solve at once.
    settings : SolverSettings
        Mesh, sweep and band counts to solve with.
    cell : Tensor
        One cell of shape (5,), the HoneycombGeometry fields in field order,
        as DesignSpace.to_cells builds it.

    Returns
    -------
    numpy.ndarray
        Frequencies in hertz, shape (n_wave_numbers, n_bands).

    Raises
    ------
    ValueError
        If the cell is not one HoneycombGeometry accepts, or if MATLAB returns
        a non-finite frequency.
    """
    geometry = HoneycombGeometry(*cell.tolist())

    # hex_side_length fills the solver's last two slots, which this project
    # keeps equal
    (frequencies,) = engines.call(
        "evaluate_dispersion",
        [[
            geometry.hex_angle, geometry.cell_width,
            geometry.face_sheet_thickness, geometry.hex_thickness,
            geometry.hex_side_length, geometry.hex_side_length,
        ]],
        settings.delta,
        settings.minimum_polynomial_order,
        settings.n_wave_numbers,
        settings.n_bands,
    )

    # A failed eigensolve returns NaN rather than raising, and blaming the
    # band structure downstream would point at the wrong step
    n_broken = int(np.sum(~np.isfinite(frequencies)))
    if n_broken:
        raise ValueError(
            f"evaluate_dispersion returned {n_broken} non-finite frequencies; "
            "the eigensolve did not converge."
        )

    return frequencies


def check_matlab(
    space: DesignSpace,
    settings: SolverSettings = SolverSettings(),
) -> np.ndarray:
    """Solve the cell at the centre of the space, so a broken setup fails fast.

    Starts its own engine and shuts it down again, so it costs one engine
    start and one solve rather than a whole run.

    Parameters
    ----------
    space : DesignSpace
        Search space whose central design, completed into a cell, is solved.
    settings : SolverSettings
        Mesh, sweep and band counts the check solves with.

    Returns
    -------
    numpy.ndarray
        Frequencies of the cell it solved, shape (n_wave_numbers, n_bands).

    Raises
    ------
    ValueError
        If the centre of the space is not buildable, or if the frequencies
        come back unusable.
    RuntimeError
        If a solver function is missing from the MATLAB path.
    """
    # The centre of the box, so the check exercises a cell the run could
    # actually propose
    design = space.bounds.mean(0)
    if not bool(space.is_feasible(design)):
        raise ValueError(
            f"The centre of the space, {design.tolist()}, is not buildable, "
            "so it cannot serve as a check; revise the bounds."
        )

    cell = space.to_cells(design)
    log.info(
        "Checking the MATLAB solver on cell %s",
        np.round(cell.numpy(), 4).tolist(),
    )

    # Timed from inside the block, so the engine's start and shutdown stay
    # out of the solve time a run will see per case
    with start_engines(1) as engines:
        engines.check()
        started = time.perf_counter()
        frequencies = solve_dispersion(engines, settings, cell)
        log.info(
            "Solved a %d x %d band structure in %.1f s, %.1f to %.1f Hz",
            *frequencies.shape, time.perf_counter() - started,
            frequencies.min(), frequencies.max(),
        )

    relative_gaps = relative_widths(frequencies)
    log.info("Relative widths %s", np.round(relative_gaps, 4).tolist())
    log.info("Widest relative gap %.6f", widest_relative_gap(relative_gaps))
    log.info("MATLAB check passed")

    return frequencies
