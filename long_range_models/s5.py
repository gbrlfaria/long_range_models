"""This module implements the models described in the paper [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .ssm import discretize_zoh, make_dplr_hippo
from .types import Array


class S5Layer(nn.Module):
    """Simplified State Space (S5) layer.

    Args:
        feature_dim: \
            The dimension of the input and the output features.
        state_dim: \
            The dimension of the state.
        dt_min: \
            Minimum value for initializing the timescale parameter.
        dt_max: \
            Maximum value for initializing the timescale parameter.
        num_blocks: \
            The number of blocks for the block-diagonal initialization of the \
            state matrix. If set to `1`, regular initialization is used.
        conj_sym: \
            Whether to enforce conjugate symmetry for the state representation.
        clip_eigs: \
            Whether to clip state matrix eigenvalues to ensure they remain negative.
        eps: \
            Small positive constant for eigenvalue clipping.

    Parameters:
        Lambda_re: \
            The real part of the diagonal state matrix.
        Lambda_im: \
            The imaginary part of the diagonal state matrix.
        B_comps: \
            The input matrix, parameterized by its real and imaginary components.
        C_comps: \
            The output matrix, parameterized by its real and imaginary components.
        D: \
            The feedthrough matrix.
        log_dt: \
            The timescale parameter.
    """

    feature_dim: int
    state_dim: int
    dt_min: float = 0.001
    dt_max: float = 0.1
    num_blocks: int = 1
    conj_sym: bool = False
    clip_eigs: bool = False
    eps: float = 1e-4

    def setup(self):
        # Variable aliasing (for convenience)
        input_dim = self.feature_dim
        output_dim = self.feature_dim
        state_dim = self.state_dim
        dt_min = self.dt_min
        dt_max = self.dt_max
        num_blocks = self.num_blocks
        conj_sym = self.conj_sym

        # Enforce conjugate symmetry
        if conj_sym:
            # In this case, we can concretely parameterize the model using
            # half the state dimension
            assert state_dim % (2 * num_blocks) == 0, (
                "to enforce conjugate symmetry, the state dimension must be divisible"
                "by two times the number of blocks"
            )
            concrete_state_dim = state_dim // 2
        else:
            assert state_dim % num_blocks == 0, (
                "the state dimension must be divisible by the number of blocks" ""
            )
            concrete_state_dim = state_dim

        # HiPPO initialization
        if num_blocks == 1:
            # Regular initialization
            Lambda, _, _, V = make_dplr_hippo(state_dim)
            Lambda = Lambda[:concrete_state_dim]
            V = V[:, :concrete_state_dim]
        else:
            # Block-diagonal initialization
            block_size = state_dim // num_blocks
            concrete_block_size = concrete_state_dim // num_blocks

            Lambda_block, _, _, V_block = make_dplr_hippo(block_size)
            Lambda_block = Lambda_block[:concrete_block_size]
            V_block = V_block[:, :concrete_block_size]

            Lambda = jnp.concatenate([Lambda_block] * num_blocks)
            V = jsp.linalg.block_diag(*([V_block] * num_blocks))

        # Parameter initializers
        def B_comps_init(key, shape):
            B = nn.initializers.lecun_normal()(key, shape)
            B = V.conj().T @ B
            # Split complex parameters into real components. This is required
            # due to JAX limitations in computing gradients of complex variables
            return jnp.stack((B.real, B.imag))

        def C_comps_init(key, shape):
            C = nn.initializers.lecun_normal()(key, shape)
            C = C @ V
            # Split complex parameters into real components. This is required
            # due to JAX limitations in computing gradients of complex variables
            return jnp.stack((C.real, C.imag))

        def log_dt_init(key, shape):
            u = nn.initializers.uniform(scale=1.0)(key, shape)
            log_dt = (jnp.log(dt_max) - jnp.log(dt_min)) * u + jnp.log(dt_min)
            return log_dt

        D_init = nn.initializers.normal(stddev=1.0)

        # Parameters
        self.Lambda_re = self.param("Lambda_re", lambda *_: Lambda.real)
        self.Lambda_im = self.param("Lambda_im", lambda *_: Lambda.imag)
        self.B_comps = self.param("B_comps", B_comps_init, (state_dim, input_dim))
        self.C_comps = self.param("C_comps", C_comps_init, (output_dim, state_dim))
        self.D = self.param("D", D_init, (input_dim,))
        self.log_dt = self.param("log_dt", log_dt_init, (concrete_state_dim,))

    def __call__(self, inputs: Array, timescale: float = 1.0) -> Array:
        """Applies the Simplified State Space (S5) layer to the input data.

        Args:
            inputs: \
                The input data with shape `(batch_size, input_length, feature_dim)`.
            timescale: \
                The time discretization scaling factor. Useful for matching the \
                sampling rate of the input.

        Returns:
            The output data with shape `(batch_size, input_length, feature_dim)`.
        """

        # Get input shape
        batch_size, input_length, _ = inputs.shape

        # Materialize parameters
        Lambda = self.Lambda_re + 1j * self.Lambda_im
        B = self.B_comps[0] + 1j * self.B_comps[1]
        C = self.C_comps[0] + 1j * self.C_comps[1]
        D = self.D

        # Apply time scaling
        dt = jnp.exp(self.log_dt) * timescale

        # Clip eigenvalues (to ensure they are negative)
        if self.clip_eigs:
            Lambda = jnp.clip(Lambda.real, None, -self.eps) + 1j * Lambda.imag

        # Discretize parameters
        Lambda_, B_ = discretize_zoh(Lambda, B, dt)
        C_, D_ = C, D

        # Apply layer
        def apply_step(element_i: Array, element_j: Array):
            # Binary operator for parallel scan of linear recurrence
            a_i, bu_i = element_i
            a_j, bu_j = element_j
            return a_j * a_i, a_j * bu_i + bu_j

        Ls = jnp.broadcast_to(Lambda_, (batch_size, input_length) + Lambda_.shape)
        Bu = jnp.einsum("PH,BLH->BLP", B_, inputs)
        _, xs = jax.lax.associative_scan(apply_step, (Ls, Bu), axis=1)

        # Because of the parametrization, we must multiply the result by two when
        # enforcing conjugate symmetry to account for the conjugate pairs
        Cx = jnp.einsum("HP,BLP->BLH", C_, xs).real * (2.0 if self.conj_sym else 1.0)
        Du = jnp.einsum("H,BLH->BLH", D_, inputs)
        ys = Cx + Du

        return ys
