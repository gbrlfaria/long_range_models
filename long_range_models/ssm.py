"""This module provides utility functions related to state space models."""

from typing import Tuple

import jax.numpy as jnp

from .types import Array


def make_hippo(N: int) -> Tuple[Array, Array]:
    """Computes a HiPPO-LegS matrix.

    Args:
        N: The size of the matrix.

    Returns:
        A tuple `(A, B)` where `A` is the HiPPO matrix and `B` is the associated input \
        vector.
    """

    indices = jnp.arange(N)
    k, n = jnp.meshgrid(indices, indices)

    L = jnp.tril((2 * n + 1) ** (1 / 2) * (2 * k + 1) ** (1 / 2), k=-1)
    A = jnp.diag(n[:, 0] + 1) + L
    B = ((2 * n[:, 0] + 1) ** (1 / 2))[:, None]

    return -A, B


def make_nplr_hippo(N: int) -> Tuple[Array, Array, Array]:
    """Computes the normal plus low-rank (NPLR) form of a HiPPO-LegS matrix.

    Args:
        N: The size of the matrix.

    Returns:
        A tuple `(S, P, B)` where `S` is the HiPPO matrix in the NPLR form, `P` is the \
        low-rank term, and `B` is the associated input vector.
    """

    A, B = make_hippo(N)
    P = jnp.sqrt(jnp.arange(N) + 1 / 2)
    S = A + P[:, None] @ P[None, :]
    return S, P, B


def make_dplr_hippo(N: int) -> Tuple[Array, Array, Array, Array]:
    """Computes the diagonal plus low-rank (DPLR) form a HiPPO-LegS matrix.

    Args:
        N: The size of the matrix.

    Returns:
        A tuple `(Lambda, P, B, V)`, where `Lambda` is the diagonalized HiPPO matrix, \
        `P` is the low-rank term, `B` is the associated input vector, and `V` is the \
        matrix containing the eigenvectors used in the diagonalization of the HiPPO \
        matrix.
    """

    S, P, B = make_nplr_hippo(N)

    # Diagonalize S to obtain V * Lambda * V^*
    # Since S is essentially a skew-symmetric matrix with a non-zero diagonal, we can
    # compute its real and complex eigenvalues separately
    Lambda_re = jnp.mean(jnp.diag(S))
    Lambda_im, V = jnp.linalg.eigh(S * -1j)
    Lambda = Lambda_re + 1j * Lambda_im

    # Conjugate matrices
    B = V.conj().T @ B
    P = V.conj().T @ P

    return Lambda, P, B, V


def discretize_zoh(Lambda: Array, B: Array, dt: Array) -> Tuple[Array, Array]:
    """Discretizes a diagonal state space model using the zero-order hold (ZOH) method.

    Args:
        Lambda: \
            The diagonalized state matrix.
        B: \
            The input matrix.
        dt: \
            The discretization timestep.
    
    Returns:
        A tuple containing the discretized state matrix (`Lambda`) and the discretized \
        input matrix (`B`).
    """

    I = jnp.ones_like(Lambda)
    Lambda_ = jnp.exp(Lambda * dt)
    B_ = ((1 / Lambda) * (Lambda_ - I))[..., None] * B
    return Lambda_, B_
