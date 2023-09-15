# Long Range Models

[![PyPI version](https://badge.fury.io/py/long_range_models.svg)](https://badge.fury.io/py/long_range_models)
[![Static Badge](https://img.shields.io/badge/powered%20by-Flax-blue)](https://github.com/google/flax)
[![Static Badge](https://img.shields.io/badge/license-MIT-yellow)](/LICENSE)


A collection of simple implementations of long-range sequence models, including [LRU](/long_range_models/lru.py), [S5](/long_range_models/s5.py), and [S4](/long_range_models/s4.py).
More implementations to come.

## Install

```bash
$ pip install long_range_models
```

## Usage

This library offers detailed documentation for every module and layer implemented.
Models are created by composing different pieces together.
Check out the examples below.

### Discrete sequence data

Consider a language model built with an LRU sequence layer and the architecture proposed in the S4 paper:

```py
from functools import partial
import jax.random as jrandom
from long_range_models import SequenceModel, S4Module, LRULayer

rng = jrandom.PRNGKey(0)

model = SequenceModel(
  num_tokens=1000,
  module=S4Module(
    sequence_layer=partial(LRULayer, state_dim=256),
    dim=128,
    depth=6,
  ),
)

x = jrandom.randint(rng, (1, 1024), 0, 1000)

variables = model.init(rng, x)
model.apply(variables, x)  # (1, 1024, 1000)

```

### Continuous sequence data

For sequences with continuous values, the setup looks as follows:

```py
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

```

**Note:** both model types offer several customization options. Make sure to check out [their documentation](/long_range_models/sequence_models.py).

## Upcoming features

- **More implementations:** Extend the library with models like S4D, S4Liquid, BiGS, Hyena, RetNet, SGConv, H3, and others.
- **Customization:** Allow users to better customize currently implemented layers and architectures (e.g., activation functions, initialization, etc.).
- **Sequential API:** Allow recurrent models to run sequentially, allowing for efficient inference.
