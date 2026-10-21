# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures for model-export tests."""

import jax
import jax.numpy as jnp
import jraph
import omegaconf
import pytest


@pytest.fixture(autouse=True)
def no_jax_preallocate(monkeypatch):
    """Prevent JAX from pre-allocating GPU memory during tests."""
    import os
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal YAML config file that resembles a Tensorial model config."""
    cfg = {
        "r_max": 5.0,
        "from_data": {
            "atomic_numbers": [1, 6, 8],
        },
        "model": {
            "model": {
                "layers": [
                    {
                        "_target_": "tensorial.mlip.calculators.ase.SomeLayer",
                    },
                    {
                        "_target_": "tensorial.mlip.calculators.ase.SomeOtherLayer",
                    },
                    {
                        "_target_": "tensorial.deploy.exporter.NodewiseReduce",
                    },
                    {
                        "_target_": "tensorial.deploy.exporter.Grad",
                    },
                    {
                        "_target_": "tensorial.deploy.exporter.CalcStresses",
                    },
                ],
            },
        },
    }
    cfg_file = tmp_path / "test_config.yaml"
    with open(cfg_file, "w") as f:
        omegaconf.OmegaConf.save(cfg, f)
    return cfg_file


@pytest.fixture
def minimal_graph():
    """Create a minimal jraph GraphsTuple for testing energy_fn."""
    n_nodes = 4
    n_edges = 6
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    graph = jraph.GraphsTuple(
        nodes={"positions": positions, "atomic_numbers": jnp.array([[6], [1], [8], [1]])},
        edges={},
        globals={},
        senders=jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0, 2, 0, 3], dtype=jnp.int32),
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([n_edges]),
    )
    return graph


@pytest.fixture
def particle_data():
    """Create a particle_data dict for testing."""
    return {
        "species": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        "valid_mask": jnp.array([True, True, True, True]),
    }


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)
