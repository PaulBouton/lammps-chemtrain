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

"""Exports sparse and dense smoothly truncated Lennard-Jones models."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")

import omegaconf
import jraph
import reax
import ase

from chemtrain.deploy import exporter, graphs
from tensorial import config as tencfg
from tensorial.mlip.calculators.ase import Calculator


class TensorialExporter(exporter.Exporter):

    def __init__(self, config, ckpt, unit_style):

        self.unit_style = unit_style

        cfg = omegaconf.OmegaConf.load(config)

        # Get the cutoff distance from the model config
        self.r_cutoff = float(cfg["r_max"])

        # Count how many nequip layers are used by the model
        n_layers = sum(
            layer["_target_"].endswith("NequipLayer")
            for layer in cfg.model.model.layers
        )
        self.nbr_order = [n_layers, n_layers]

        # Retrieve Atomic numbers from the config
        atomic_numbers = list(cfg.from_data.atomic_numbers)
        self.atomic_numbers = jnp.asarray(atomic_numbers)

        # Keep the model only up to the per-atom energy
        # Drop NodewiseReduce, Grad, CalcStresses as they are handle
        # by the Chemtrain deploy module.
        cfg.model.model.layers = cfg.model.model.layers[:-3]

        # Load the model and the checkpoint weigths
        self.model = tencfg.instantiate(cfg.model.model)
        ckpt_data = reax.training.get_default_checkpointing().load(ckpt)

        params = ckpt_data[reax.training._checkpointing.PARAMS]
        self.params = jax.tree_util.tree_map(jnp.asarray, params)

    def energy_fn(self, position, particle_data, graph, comm=None):
        n = position.shape[0]
        g = jraph.GraphsTuple(nodes={"positions": position,"atomic_numbers": self.atomic_numbers[particle_data["species"]][:, None]},
                              edges={},
                              globals={}, #graph-level features
                              senders=graph.senders,
                              receivers=graph.receivers,
                              n_node=jnp.array([n]),
                              n_edge=jnp.array([graph.senders.shape[0]]))
        out = self.model.apply(self.params, g)

        e = out.nodes["energy"]
        e = getattr(e, "array", e)   # unwrap IrrepsArray -> jax array
        e = e[:, 0]                  # (n, 1) -> (n,)
        return jnp.where(particle_data["valid_mask"], e, 0.0)


def export_model(output_directory, config, ckpt, unit_style):
    """Exports Tensorial model with CPU & GPU (cuda) backend."""
    output_directory.mkdir(parents=True, exist_ok=True)
    model = TensorialExporter(config, ckpt, unit_style)
    model.export(platforms=("cpu", "cuda"))
    style = str(unit_style).strip("'\"")
    model.save(output_directory / f"tensorial-{style}.ptb")

def parse_args():
     parser = argparse.ArgumentParser(description="Run with a config and a checkpoint files.")
     parser.add_argument("config",
                         type=Path,
                         help="YAML config file")
     parser.add_argument("ckpt",
                         type=Path,
                         help="checkpoint file")
     parser.add_argument("unit_style",
                         type=lambda s: s.strip().strip("'\""),
                         help="LAMMPS unit type")
     return parser.parse_args()

def main():
    """Export model."""
    args = parse_args()
    export_model(Path.cwd(), args.config, args.ckpt, args.unit_style)

if __name__ == "__main__":
    main()
