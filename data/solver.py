
# ThermoTrace — FDM Solver
# Solves the 2D steady-state heat conduction (Poisson) equation:
#
#     ∇²T = -Q(x,y) / k
#
# using the Finite Difference Method with boundary conditions.
# This is the same equation ANSYS solves internally for steady-state thermal.


import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from config import GRID_SIZE, PLATE_SIZE_M, T_BOUNDARY, K_THERMAL


def build_fdm_matrix(grid: int, h: float) -> lil_matrix:
    """
    Build the sparse coefficient matrix A for the 2D Poisson equation
    using a 5-point finite difference stencil.

    Interior node equation:
        (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1] - 4*T[i,j]) / h² = -Q[i,j]/k

    Boundary nodes are handled conditions (fixed T = T_BOUNDARY).

    Args:
        grid : number of grid points per side (including boundaries)
        h    : grid spacing (meters)

    Returns:
        A    : sparse matrix of shape (grid², grid²)
    """
    N = grid * grid
    A = lil_matrix((N, N))

    def idx(i, j):
        return i * grid + j

    for i in range(grid):
        for j in range(grid):
            node = idx(i, j)

            # Boundary nodes — enforce T = T_BOUNDARY (Dirichlet)
            if i == 0 or i == grid - 1 or j == 0 or j == grid - 1:
                A[node, node] = 1.0

            # Interior nodes — 5-point Laplacian stencil
            else:
                A[node, idx(i - 1, j)] =  1.0
                A[node, idx(i + 1, j)] =  1.0
                A[node, idx(i, j - 1)] =  1.0
                A[node, idx(i, j + 1)] =  1.0
                A[node, node]          = -4.0

    return A.tocsr()


def solve_temperature(Q: np.ndarray,
                      grid: int       = GRID_SIZE,
                      h: float        = PLATE_SIZE_M / (GRID_SIZE - 1),
                      T_bc: float     = T_BOUNDARY,
                      k: float        = K_THERMAL) -> np.ndarray:
    """
    Given a heat source field Q (W/m³), solve for the steady-state
    temperature distribution T (K) over a 2D plate.

    Args:
        Q    : (grid × grid) array of volumetric heat source intensities
        grid : grid resolution
        h    : grid spacing (m)
        T_bc : boundary temperature (K)
        k    : thermal conductivity (W/m·K)

    Returns:
        T    : (grid × grid) steady-state temperature field (K)
    """
    N = grid * grid
    A = build_fdm_matrix(grid, h)

    # Build RHS vector b
    b = np.zeros(N)

    for i in range(grid):
        for j in range(grid):
            node = i * grid + j

            if i == 0 or i == grid - 1 or j == 0 or j == grid - 1:
                b[node] = T_bc                          # Dirichlet BC
            else:
                b[node] = -(Q[i, j] / k) * h ** 2      # Source term

    # Solve the sparse linear system A·T_flat = b
    T_flat = spsolve(A, b)
    T = T_flat.reshape((grid, grid))

    return T


def compute_pde_residual(T: np.ndarray,
                         Q: np.ndarray,
                         h: float  = PLATE_SIZE_M / (GRID_SIZE - 1),
                         k: float  = K_THERMAL) -> np.ndarray:
    """
    Compute the PDE residual: ∇²T + Q/k at each interior node.
    Used as the physics-informed loss term during model training.

    Residual should be ~0 everywhere if T and Q are physically consistent.

    Args:
        T : (grid × grid) temperature field
        Q : (grid × grid) heat source field
        h : grid spacing
        k : thermal conductivity

    Returns:
        residual : (grid-2 × grid-2) PDE residual at interior nodes
    """
    # Finite difference Laplacian at interior nodes
    laplacian = (
        T[:-2, 1:-1] + T[2:, 1:-1] +
        T[1:-1, :-2] + T[1:-1, 2:] -
        4 * T[1:-1, 1:-1]
    ) / h ** 2

    source_term = Q[1:-1, 1:-1] / k
    residual = laplacian + source_term

    return residual
