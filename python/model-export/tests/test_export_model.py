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

"""Tests for export-model.py."""

import importlib.util
import pathlib
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import jraph
import omegaconf
import pytest


# ---------------------------------------------------------------------------
# Load the module under test dynamically so we can mock its heavy deps
# ---------------------------------------------------------------------------
_SCRIPT_PATH = (
    pathlib.Path(__file__).parent.parent / "export-model.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "export_model", _SCRIPT_PATH,
)


def _setup_mocks():
    """Set up all mocks needed before loading the module under test.

    Returns (mocks dict, tencfg_mock).
    """
    # --- Mock reax ---
    reax_mock = mock.MagicMock()
    training_mock = mock.MagicMock()
    mock_ckpt = {"params": {"layer_0": {"weight": jnp.zeros((10, 10))}}}
    training_mock.get_default_checkpointing.return_value = mock.MagicMock(
        load=mock.MagicMock(return_value=mock_ckpt),
    )
    training_mock._checkpointing = mock.MagicMock()
    training_mock._checkpointing.PARAMS = "params"
    reax_mock.training = training_mock
    sys.modules["reax"] = reax_mock

    # --- Mock tensorial and tensorial.config ---
    mock_model = mock.MagicMock()
    # When energy_fn calls model.apply(...), it gets a result whose
    # .nodes["energy"] should be a real JAX array (shape (4, 1)).
    mock_result = mock.MagicMock()
    mock_result.nodes.__getitem__.return_value = jnp.zeros((4, 1))
    mock_model.apply.return_value = mock_result

    tencfg_mock = mock.MagicMock()
    tencfg_mock.instantiate.return_value = mock_model
    tensorial_mock = mock.MagicMock()
    tensorial_mock.config = tencfg_mock  # attribute lookup returns the right mock
    ase_mock = mock.MagicMock()
    sys.modules["tensorial"] = tensorial_mock
    sys.modules["tensorial.config"] = tencfg_mock
    sys.modules["tensorial.mlip.calculators.ase"] = ase_mock

    return {
        "mock_model": mock_model,
        "mock_result": mock_result,
    }, tencfg_mock


# ---------------------------------------------------------------------------
# Unit tests for parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:

    def test_parse_args_returns_paths_and_unit_style(
        self, tmp_path, monkeypatch,
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        ckpt_file = tmp_path / "ckpt.pt"
        ckpt_file.write_text("")

        mod = importlib.util.module_from_spec(_SPEC)
        sys.modules["export_model"] = mod
        _SPEC.loader.exec_module(mod)

        monkeypatch.setattr(
            sys, "argv",
            ["export-model.py", str(config_file), str(ckpt_file), "lj"],
        )
        args = mod.parse_args()

        assert args.config == config_file
        assert args.ckpt == ckpt_file
        assert args.unit_style == "lj"

    @pytest.mark.parametrize(
        "unit_style_input, expected",
        [
            ("lj", "lj"),
            ("'real'", "real"),
            ('"metal"', "metal"),
            ("  real  ", "real"),
        ],
    )
    def test_unit_style_strip_quotes(
        self, tmp_path, monkeypatch, unit_style_input, expected,
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        ckpt_file = tmp_path / "ckpt.pt"
        ckpt_file.write_text("")

        mod = importlib.util.module_from_spec(_SPEC)
        sys.modules["export_model"] = mod
        _SPEC.loader.exec_module(mod)

        monkeypatch.setattr(
            sys, "argv",
            ["export-model.py", str(config_file), str(ckpt_file),
             unit_style_input],
        )
        args = mod.parse_args()
        assert args.unit_style == expected


# ---------------------------------------------------------------------------
# Unit tests for TensorialExporter
# ---------------------------------------------------------------------------


class TestTensorialExporter:

    def _make_mock_module(self, tmp_path, cfg, n_layers=3):
        """Set up mock modules and return the loaded module under test."""
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        mocks, tencfg_mock = _setup_mocks()

        # The actual import of the module
        mod = importlib.util.module_from_spec(_SPEC)
        sys.modules["export_model"] = mod
        _SPEC.loader.exec_module(mod)

        return cfg_file, mocks, mod, tencfg_mock

    def test_init_sets_r_cutoff(self, tmp_path, minimal_config):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        cfg["r_max"] = 5.5
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        cfg_file, mocks, mod, _ = self._make_mock_module(tmp_path, cfg)

        exporter = mod.TensorialExporter(
            str(cfg_file), "dummy_ckpt", "real",
        )
        assert exporter.r_cutoff == 5.5

    def test_init_sets_n_layers_from_nequip_layers(self, tmp_path, minimal_config):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        # Replace the non-Nequip layers with Nequip ones
        cfg["model"]["model"]["layers"] = [
            {"_target_": "tensorial.mlip.calculators.ase.SomeNequipLayer"},
            {"_target_": "tensorial.mlip.calculators.ase.SomeNequipLayer"},
            {"_target_": "tensorial.deploy.exporter.NodewiseReduce"},
            {"_target_": "tensorial.deploy.exporter.Grad"},
            {"_target_": "tensorial.deploy.exporter.CalcStresses"},
        ]
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        cfg_file, mocks, mod, _ = self._make_mock_module(
            tmp_path, cfg,
        )

        exporter = mod.TensorialExporter(
            str(cfg_file), "dummy_ckpt", "real",
        )
        # Count of Nequip layers determines nbr_order
        assert exporter.nbr_order == [2, 2]

    def test_init_sets_atomic_numbers(self, tmp_path, minimal_config):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        cfg["from_data"]["atomic_numbers"] = [1, 6, 8, 16]
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        cfg_file, mocks, mod, _ = self._make_mock_module(
            tmp_path, cfg,
        )

        exporter = mod.TensorialExporter(
            str(cfg_file), "dummy_ckpt", "real",
        )
        assert list(exporter.atomic_numbers) == [1, 6, 8, 16]

    def test_init_strips_last_three_layers(self, tmp_path, minimal_config):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        # Add 4 Nequip layers
        cfg["model"]["model"]["layers"] = (
            [
                {"_target_": "tensorial.mlip.calculators.ase.SomeNequipLayer"}
                for _ in range(4)
            ]
            + [
                {"_target_": "tensorial.deploy.exporter.NodewiseReduce"},
                {"_target_": "tensorial.deploy.exporter.Grad"},
                {"_target_": "tensorial.deploy.exporter.CalcStresses"},
            ]
        )
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        cfg_file, mocks, mod, tencfg_mock = self._make_mock_module(tmp_path, cfg)

        mod.TensorialExporter(str(cfg_file), "dummy_ckpt", "real")
        # tencfg.instantiate should have been called with the config
        # that only contains the Nequip layers (not the last 3)
        # The config is: cfg.model.model.layers (OmegaConf passes cfg.model.model)
        call_args = tencfg_mock.instantiate.call_args
        layers_arg = call_args[0][0]["layers"]
        # 4 Nequip + 3 post-processing = 7 total, should keep only 4
        assert len(layers_arg) == 4

    def test_energy_fn_returns_correct_shape(
        self, tmp_path, minimal_config, minimal_graph, particle_data,
    ):
        """Use minimal_config, minimal_graph, and particle_data fixtures
        from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        _, mocks, mod, _ = self._make_mock_module(tmp_path, cfg)

        # Configure the mock to return a proper JAX array
        energy_array = jnp.zeros((4, 1))
        mocks["mock_result"].nodes.__getitem__.return_value = energy_array

        exporter = mod.TensorialExporter(
            str(cfg_file), "dummy_ckpt", "real",
        )

        energy = exporter.energy_fn(
            minimal_graph.nodes["positions"],
            particle_data,
            minimal_graph,
        )
        assert energy.shape == (4,)
        assert jnp.all(jnp.isfinite(energy))

    def test_energy_fn_masks_invalid_particles(
        self, tmp_path, minimal_config, minimal_graph,
    ):
        """Use minimal_config and minimal_graph fixtures from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        _, mocks, mod, _ = self._make_mock_module(tmp_path, cfg)

        # Configure the mock to return a proper JAX array with non-zero energy
        energy_array = jnp.array(
            [[1.0], [2.0], [3.0], [4.0]], dtype=jnp.float32,
        )
        mocks["mock_result"].nodes.__getitem__.return_value = energy_array

        exporter = mod.TensorialExporter(
            str(cfg_file), "dummy_ckpt", "real",
        )

        # Reuse positions from the fixture
        positions = minimal_graph.nodes["positions"]
        particle_data = {
            "species": jnp.array([0, 1, 2, 0], dtype=jnp.int32),
            "valid_mask": jnp.array([True, True, False, True]),
        }
        graph = jraph.GraphsTuple(
            nodes={"positions": positions,
                   "atomic_numbers": jnp.array([[6], [1], [8], [1]])},
            edges={},
            globals={},
            senders=jnp.array([0, 1, 2], dtype=jnp.int32),
            receivers=jnp.array([1, 2, 3], dtype=jnp.int32),
            n_node=jnp.array([4]),
            n_edge=jnp.array([3]),
        )

        energy = exporter.energy_fn(positions, particle_data, graph)
        # Invalid particle (index 2) should have zero energy
        assert energy[2] == jnp.float32(0.0)


# ---------------------------------------------------------------------------
# Unit tests for export_model function
# ---------------------------------------------------------------------------


class TestExportModelFunction:

    def _mock_export_model_call(self, tmp_path, cfg, output_dir,
                                unit_style="lj"):
        """Set up mocks and return (mod, exporter_mock) for export_model
        tests."""
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        ckpt_file = tmp_path / "ckpt.pt"
        ckpt_file.write_text("")

        _setup_mocks()

        mod = importlib.util.module_from_spec(_SPEC)
        sys.modules["export_model"] = mod
        _SPEC.loader.exec_module(mod)

        # Build a mock exporter instance with the expected interface
        exporter_mock = mock.MagicMock()
        exporter_mock.r_cutoff = 5.0
        exporter_mock.nbr_order = [2, 2]
        exporter_mock.atomic_numbers = jnp.array([1, 6, 8])
        exporter_mock.model = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_result.nodes.__getitem__.return_value = jnp.zeros((4, 1))
        exporter_mock.model.apply.return_value = mock_result
        exporter_mock.export = mock.MagicMock()
        exporter_mock.save = mock.MagicMock()

        # Replace the class in the module namespace so `TensorialExporter(...)`
        # returns our mock instance and the instance methods work correctly.
        mod.TensorialExporter = mock.MagicMock(return_value=exporter_mock)

        return mod, cfg_file, ckpt_file, output_dir, exporter_mock

    def test_export_model_creates_output_directory(
        self, tmp_path, minimal_config,
    ):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, output_dir, exporter_instance = \
            self._mock_export_model_call(tmp_path, cfg, output_dir)

        mod.export_model(output_dir, cfg_file, ckpt_file, "lj")

        assert output_dir.exists()
        # verify that TensorialExporter was instantiated
        mod.TensorialExporter.assert_called_once()

    def test_export_model_saves_protobuf_file(
        self, tmp_path, minimal_config,
    ):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, output_dir, exporter_instance = \
            self._mock_export_model_call(tmp_path, cfg, output_dir)

        mod.export_model(output_dir, cfg_file, ckpt_file, "lj")

        # Check that save was called with the expected filename
        exporter_instance.save.assert_called_once()
        save_path = exporter_instance.save.call_args[0][0]
        assert "tensorial-lj.ptb" in str(save_path)

    @pytest.mark.parametrize(
        "unit_style, expected_filename",
        [
            ("lj", "tensorial-lj.ptb"),
            ("real", "tensorial-real.ptb"),
            ("'real'", "tensorial-real.ptb"),
            ('"metal"', "tensorial-metal.ptb"),
        ],
    )
    def test_export_model_filename_from_unit_style(
        self, tmp_path, minimal_config, unit_style, expected_filename,
    ):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, output_dir, exporter_instance = \
            self._mock_export_model_call(tmp_path, cfg, output_dir,
                                         unit_style)

        mod.export_model(output_dir, cfg_file, ckpt_file, unit_style)

        save_path = exporter_instance.save.call_args[0][0]
        assert expected_filename in str(save_path)

    @pytest.mark.parametrize(
        "unit_style, expected_str",
        [
            ("lj", "lj"),
            ("'real'", "real"),
            ('"metal"', "metal"),
            ("  real  ", "real"),
        ],
    )
    def test_unit_style_parsing(self, unit_style, expected_str):
        """Verify parse_args strips quotes and whitespace."""
        _setup_mocks()
        mod = importlib.util.module_from_spec(_SPEC)
        sys.modules["export_model"] = mod
        _SPEC.loader.exec_module(mod)

        with mock.patch.object(
            sys, "argv",
            ["export-model.py", "/tmp/c.yaml", "/tmp/k.pt", unit_style],
        ):
            args = mod.parse_args()
        assert args.unit_style == expected_str


# ---------------------------------------------------------------------------
# Unit tests for main()
# ---------------------------------------------------------------------------


class TestMain:

    def test_main_calls_export_model_with_cwd(
        self, tmp_path, monkeypatch, minimal_config,
    ):
        """Use minimal_config fixture from conftest.py."""
        cfg = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(minimal_config),
        )
        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            omegaconf.OmegaConf.save(cfg, f)

        ckpt_file = tmp_path / "ckpt.pt"
        ckpt_file.write_text("")

        _setup_mocks()

        mod = importlib.util.module_from_spec(_SPEC)
        sys.modules["export_model"] = mod
        _SPEC.loader.exec_module(mod)

        exporter_mock = mock.MagicMock()
        exporter_mock.r_cutoff = 5.0
        exporter_mock.nbr_order = [2, 2]
        exporter_mock.atomic_numbers = jnp.array([1, 6, 8])
        exporter_mock.model = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_result.nodes.__getitem__.return_value = jnp.zeros((4, 1))
        exporter_mock.model.apply.return_value = mock_result
        exporter_mock.export = mock.MagicMock()
        exporter_mock.save = mock.MagicMock()

        # Replace the class in the module namespace
        mod.TensorialExporter = mock.MagicMock(return_value=exporter_mock)

        monkeypatch.setattr(sys, "argv", [
            "export-model.py",
            str(cfg_file),
            str(ckpt_file),
            "lj",
        ])
        monkeypatch.chdir(tmp_path)

        mod.main()

        # export_model should have been called with cwd = tmp_path
        exporter_mock.save.assert_called_once()
