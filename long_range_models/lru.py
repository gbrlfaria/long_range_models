"""This module implements the models described in the paper [Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349)."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom

from .types import Array


class LRULayer(nn.Module):
    """Linear Recurrent Unit (LRU) layer.

    Args:
        feature_dim: \
            The dimension of the input and the output features.
        state_dim: \
            The dimension of the state.
        r_min: \
            The minimum radius of the ring for sampling state matrix \
            eigenvalues in the complex plane.
        r_max: \
            The maximum radius of the ring for sampling state matrix \
            eigenvalues in the complex plane.
        max_phase: \
            The maximum phase value for eigenvalue initialization.

    Parameters:
        nu_log: \
            The magnitude parameter of the diagonal state matrix.
        theta_log: \
            The phase parameter of the diagonal state matrix.
        B_re: \
            The real part of the input matrix.
        B_im: \
            The imaginary part of the input matrix.
        C_re: \
            The real part of the output matrix.
        C_im: \
            The imaginary part of the output matrix.
        D: \
            The feedthrough matrix.
        gamma_log: \
            The input normalization parameter.
    """

    feature_dim: int
    state_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28

    def setup(self):
        # Variable aliasing (for convenience)
        input_dim = self.feature_dim
        output_dim = self.feature_dim
        state_dim = self.state_dim
        r_min = self.r_min
        r_max = self.r_max
        max_phase = self.max_phase

        # Parameter initializers
        def nu_log_init(key, shape):
            u1 = jrandom.uniform(key, shape)
            nu = -0.5 * (jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
            return jnp.log(nu)

        def theta_log_init(key, shape):
            u2 = jrandom.uniform(key, shape)
            theta = max_phase * u2
            return jnp.log(theta)

        B_init = nn.initializers.variance_scaling(0.5, "fan_out", "normal")
        C_init = nn.initializers.variance_scaling(1.0, "fan_out", "normal")
        D_init = nn.initializers.normal(stddev=1.0)

        # Parameters
        self.nu_log = self.param("nu_log", nu_log_init, (state_dim,))
        self.theta_log = self.param("theta_log", theta_log_init, (state_dim,))
        self.B_re = self.param("B_re", B_init, (state_dim, input_dim))
        self.B_im = self.param("B_im", B_init, (state_dim, input_dim))
        self.C_re = self.param("C_re", C_init, (output_dim, state_dim))
        self.C_im = self.param("C_im", C_init, (output_dim, state_dim))
        self.D = self.param("D", D_init, (output_dim,))

        def init_gamma_log(*_):
            diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
            gamma = jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2)
            return jnp.log(gamma)

        self.gamma_log = self.param("gamma_log", init_gamma_log)

    def __call__(self, inputs: Array) -> Array:
        """Applies the Linear Recurrent Unit (LRU) layer to the input data.

        Args:
            inputs: \
                The input data with shape `(batch_size, input_length, feature_dim)`.

        Returns:
            The output data with shape `(batch_size, input_length, feature_dim)`.
        """

        # Get input shape
        batch_size, input_length, _ = inputs.shape

        # Materialize parameters
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B = self.B_re + 1j * self.B_im
        C = self.C_re + 1j * self.C_im
        D = self.D
        gamma = jnp.exp(self.gamma_log)

        # Apply normalization
        B_norm = B * jnp.expand_dims(gamma, axis=-1)

        # Apply layer
        def apply_step(element_i: Array, element_j: Array):
            # Binary operator for parallel scan of linear recurrence
            a_i, bu_i = element_i
            a_j, bu_j = element_j
            return a_j * a_i, a_j * bu_i + bu_j

        Ls = jnp.broadcast_to(Lambda, (batch_size, input_length, self.state_dim))
        Bu = jnp.einsum("PH,BLH->BLP", B_norm, inputs)
        _, xs = jax.lax.associative_scan(apply_step, (Ls, Bu), axis=1)

        Cx = jnp.einsum("HP,BLP->BLH", C, xs).real
        Du = jnp.einsum("H,BLH->BLH", D, inputs)
        ys = Cx + Du

        return ys
