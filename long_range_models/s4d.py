import flax.linen as nn
import jax.numpy as jnp

from .ssm import make_dplr_hippo
from .types import Array


class S4DLayer(nn.Module):
    feature_dim: int
    state_dim: int
    mix_outputs: bool = True
    tie_weights: bool = False
    conj_sym: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    init: str = "lin"  # lin, inv, hippo
    clip_eigs: bool = True
    eps: float = 1e-4

    def setup(self):
        # Variable aliasing (for convenience)
        feature_dim = self.feature_dim
        state_dim = self.state_dim
        tie_weights = self.tie_weights
        conj_sym = self.conj_sym
        dt_min = self.dt_min
        dt_max = self.dt_max
        init = self.init

        # TODO: Document, enforced conjugate symmetry
        if conj_sym:
            assert state_dim % 2 == 0, "The state dimension must be divisible by 2"
            concrete_state_dim = state_dim // 2
        else:
            concrete_state_dim = state_dim

        # TODO: Comment
        if tie_weights:
            feature_dim = 1

        # TODO: allow custom initialization

        # Parameter initializers
        def Lambda_comps_init(key, shape, full_shape):
            (H, N) = shape
            (N_full,) = full_shape

            if init == "lin":
                Lambda_re = nn.initializers.constant(-0.5)(key, shape)
                Lambda_im = jnp.pi * jnp.arange(N)
            elif init == "inv":
                Lambda_re = nn.initializers.constant(-0.5)(key, shape)
                Lambda_im = (N / jnp.pi) * (N / (2.0 * jnp.arange(N) + 1.0) - 1.0)
            elif init == "hippo":
                Lambda, _, _, _ = make_dplr_hippo(N_full)
                Lambda_re = Lambda.real[:N]
                Lambda_im = Lambda.imag[:N]
            else:
                raise ValueError(f"Unknown initialization mode '{init}'")

            Lambda_re = jnp.broadcast_to(Lambda_re, shape).copy()
            Lambda_im = jnp.broadcast_to(Lambda_im, shape).copy()

            # Split complex parameters into real components. This is required
            # due to JAX limitations in computing gradients of complex variables
            return jnp.stack((Lambda_re, Lambda_im))

        def C_comps_init(key, shape):
            C_re = nn.initializers.normal(stddev=1.0)(key, shape)
            C_im = nn.initializers.normal(stddev=1.0)(key, shape)
            # Split complex parameters into real components. This is required
            # due to JAX limitations in computing gradients of complex variables
            return jnp.stack((C_re, C_im))

        def log_dt_init(key, shape):
            u = nn.initializers.uniform(scale=1.0)(key, shape)
            log_dt = (jnp.log(dt_max) - jnp.log(dt_min)) * u + jnp.log(dt_min)
            return log_dt

        D_init = nn.initializers.normal(stddev=1.0)

        # Parameters
        self.Lambda_comps = self.param(
            "Lambda_comps",
            Lambda_comps_init,
            (feature_dim, concrete_state_dim),
            (state_dim,),
        )
        self.C_comps = self.param(
            "C_comps", C_comps_init, (feature_dim, concrete_state_dim)
        )
        self.D = self.param("D", D_init, (feature_dim,))
        self.log_dt = self.param("log_dt", log_dt_init, (feature_dim,))

    def __call__(self, inputs: Array, timescale: float = 1.0) -> Array:
        # Get input shape
        _, input_length, _ = inputs.shape

        # Materialize parameters
        Lambda = self.Lambda_comps[0] + 1j * self.Lambda_comps[1]
        C = self.C_comps[0] + 1j * self.C_comps[1]
        D = self.D

        # Apply time scaling
        dt = jnp.exp(self.log_dt) * timescale

        # Clip eigenvalues (to ensure they are negative)
        if self.clip_eigs:
            Lambda = jnp.clip(Lambda.real, None, -self.eps) + 1j * Lambda.imag

        # If weights are tied, broadcast them across the feature dimension
        if self.tie_weights:
            Lambda = jnp.broadcast_to(
                Lambda,
                (self.feature_dim, Lambda.shape[-1]),
            ).copy()
            C = jnp.broadcast_to(C, (self.feature_dim, C.shape[-1])).copy()
            D = jnp.broadcast_to(D, (self.feature_dim,)).copy()

        # Vandermonde kernel computation with ZOH discretization
        Lambda_dt = Lambda * dt[..., None]
        Lambda_ = jnp.exp(Lambda_dt)
        B_ = (1.0 / Lambda) * (Lambda_ - 1.0)
        K = jnp.exp(Lambda_dt[None, ...] * jnp.arange(input_length)[..., None, None])
        K = jnp.einsum("HP,LHP->LH", B_ * C, K).real * (2.0 if self.conj_sym else 1.0)

        # Convolution
        L = input_length
        k_f = jnp.fft.rfft(K, n=2 * L, axis=0)
        u_f = jnp.fft.rfft(inputs, n=2 * L, axis=1)
        ys = jnp.fft.irfft(k_f * u_f, n=2 * L, axis=1)[:, :L, ...]

        # Apply feedthrough matrix
        ys = ys + D * inputs

        if self.mix_outputs:
            ys = nn.Dense(self.feature_dim)(ys)

        return ys
