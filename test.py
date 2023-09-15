from functools import partial
import jax.random as jrandom
from long_range_models import ContinuousSequenceModel, S4Module, LRULayer

rng = jrandom.PRNGKey(0)

model = ContinuousSequenceModel(
  out_dim=10,
  module=S4Module(
    sequence_layer=partial(LRULayer, state_dim=256),
    dim=128,
    depth=6,
  ),
)

x = jrandom.normal(rng, (1, 1024, 32))

variables = model.init(rng, x)
model.apply(variables, x)  # (1, 1024, 10)
