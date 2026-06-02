# =============================================================================
# ThermoTrace — Data Generation Pipeline
#
# For each sample:
#   1. Randomly place 1–3 circular heat sources on the plate
#   2. Solve ∇²T = -Q/k using FDM solver
#   3. Optionally add Gaussian noise to T (simulates real sensor noise)
#   4. Save (T_noisy, Q, source_params) triplets as compressed .npz files
#
# Dataset split: 8000 train / 1000 val / 1000 test
# =============================================================================

import numpy as np
import os
import time
from tqdm import tqdm

from solver import solve_temperature
from config import (
    GRID_SIZE, MIN_SOURCES, MAX_SOURCES,
    MIN_INTENSITY, MAX_INTENSITY,
    MIN_RADIUS_CELLS, MAX_RADIUS_CELLS,
    SOURCE_MARGIN, ADD_NOISE, NOISE_STD_K,
    NUM_TRAIN, NUM_VAL, NUM_TEST,
    DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE
)


def place_heat_sources(grid: int = GRID_SIZE,
                       num_sources: int = None,
                       rng: np.random.Generator = None) -> tuple[np.ndarray, list]:
    """
    Randomly place circular heat sources on the grid.

    Args:
        grid        : grid resolution
        num_sources : number of sources (random if None)
        rng         : numpy random generator (for reproducibility)

    Returns:
        Q           : (grid × grid) heat source field (W/m³)
        sources     : list of dicts with source metadata
                      [{'cx', 'cy', 'radius', 'intensity'}, ...]
    """
    if rng is None:
        rng = np.random.default_rng()

    n = num_sources if num_sources else rng.integers(MIN_SOURCES, MAX_SOURCES + 1)

    Q = np.zeros((grid, grid), dtype=np.float32)
    sources = []

    # Pre-build coordinate grids for vectorized distance computation
    yy, xx = np.mgrid[0:grid, 0:grid]

    for _ in range(n):
        cx = int(rng.integers(SOURCE_MARGIN, grid - SOURCE_MARGIN))
        cy = int(rng.integers(SOURCE_MARGIN, grid - SOURCE_MARGIN))
        radius = int(rng.integers(MIN_RADIUS_CELLS, MAX_RADIUS_CELLS + 1))
        intensity = float(rng.uniform(MIN_INTENSITY, MAX_INTENSITY))

        # Vectorized circular mask
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        Q[mask] += intensity        # Sources can overlap — intensities add up

        sources.append({
            'cx': cx, 'cy': cy,
            'radius': radius,
            'intensity': intensity
        })

    return Q, sources


def generate_sample(rng: np.random.Generator = None,
                    add_noise: bool = ADD_NOISE) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Generate a single (T, Q, sources) training sample.

    Args:
        rng       : numpy random generator
        add_noise : whether to add Gaussian noise to temperature field

    Returns:
        T_input   : (grid × grid) temperature field, possibly noisy (model input)
        Q_target  : (grid × grid) heat source field (model target / ground truth)
        sources   : list of source parameter dicts (metadata)
    """
    if rng is None:
        rng = np.random.default_rng()

    Q, sources = place_heat_sources(rng=rng)
    T = solve_temperature(Q).astype(np.float32)

    T_input = T.copy()
    if add_noise:
        noise = rng.normal(0, NOISE_STD_K, T.shape).astype(np.float32)
        T_input = T + noise

    return T_input, Q, sources


def generate_dataset(num_samples: int,
                     seed: int = None,
                     desc: str = "Generating") -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a full dataset of (T, Q) pairs.

    Args:
        num_samples : number of samples to generate
        seed        : random seed for reproducibility
        desc        : progress bar label

    Returns:
        T_data : (N × 64 × 64) array of temperature fields
        Q_data : (N × 64 × 64) array of heat source fields
    """
    rng = np.random.default_rng(seed)

    T_data = np.zeros((num_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    Q_data = np.zeros((num_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for i in tqdm(range(num_samples), desc=desc, unit="sample"):
        T_input, Q_target, _ = generate_sample(rng=rng)
        T_data[i] = T_input
        Q_data[i] = Q_target

    return T_data, Q_data


def normalize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalize array to [0, 1]. Returns normalized array + stats."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8), mn, mx


def run_data_generation():
    """
    Full data generation pipeline.
    Generates train/val/test splits and saves as compressed .npz files.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    total = NUM_TRAIN + NUM_VAL + NUM_TEST

    print(f"\n{'='*55}")
    print(f"  ThermoTrace — Dataset Generation")
    print(f"  Total samples : {total:,}  ({NUM_TRAIN} train / {NUM_VAL} val / {NUM_TEST} test)")
    print(f"  Grid size     : {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Noise         : {'ON  (σ = ' + str(NOISE_STD_K) + ' K)' if ADD_NOISE else 'OFF'}")
    print(f"{'='*55}\n")

    start = time.time()

    # Generate with fixed seeds for reproducibility (important for publication)
    splits = [
        (NUM_TRAIN, 42,  "Train",      TRAIN_FILE),
        (NUM_VAL,   123, "Validation", VAL_FILE),
        (NUM_TEST,  999, "Test",       TEST_FILE),
    ]

    for n_samples, seed, desc, filename in splits:
        T_data, Q_data = generate_dataset(n_samples, seed=seed, desc=desc)

        # Compute and store global normalization statistics for T
        # Q is kept in raw W/m³ (logged in paper for reproducibility)
        T_min, T_max = T_data.min(), T_data.max()
        Q_min, Q_max = Q_data.min(), Q_data.max()

        filepath = os.path.join(DATA_DIR, filename)
        np.savez_compressed(
            filepath,
            T=T_data, Q=Q_data,
            T_min=T_min, T_max=T_max,
            Q_min=Q_min, Q_max=Q_max
        )

        size_mb = os.path.getsize(filepath) / 1e6
        print(f"  Saved {filepath}  [{size_mb:.1f} MB]  T∈[{T_min:.1f}, {T_max:.1f}] K\n")

    elapsed = time.time() - start
    print(f"Done. Total time: {elapsed:.1f}s")
    print(f"Dataset saved to ./{DATA_DIR}/\n")


if __name__ == "__main__":
    run_data_generation()
