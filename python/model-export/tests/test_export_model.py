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
import types
from unittest import mock

import jax.numpy as jnp
import jraph
import omegaconf
import pytest


# ---------------------------------------------------------------------------
# Load the module under test dynamically so we can mock its heavy deps
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPT_PATH = next(
    (
        p
        for p in (_HERE / "export-model.py", _HERE.parent / "export-model.py")
        if p.exists()
    ),
    _HERE.parent / "export-model.py",
)
_SPEC = importlib.util.spec_from_file_location("export_model", _SCRIPT_PATH)

_DEFAULT_LAYERS_TO_STRIP = 3


class _FakeBaseExporter:
    """Stand-in for chemtrain.deploy.exporter.Exporter.

    The module under test subclasses this, so it must be a real class.
    """

    def export(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """Undo the sys.modules mocking after every test."""
    with mock.patch.dict(sys.modules):
        yield


def _setup_mocks():
    """Set up all mocks needed before loading the module under test.

    Returns a dict with the mocks tests may want to inspect/configure.
    """
    # --- Mock reax ---
    reax_mock = mock.MagicMock()
    training_mock = mock.MagicMock()
    mock_ckpt = {"params": {"layer_0": {"weight": jnp.zeros((10, 10))}}}
    ckpt_loader = mock.MagicMock(return_value=mock_ckpt)
    training_mock.get_default_checkpointing.return_value = mock.MagicMock(
        load=ckpt_loader,
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
    tensorial_mock.config = tencfg_mock
    sys.modules["tensorial"] = tensorial_mock
    sys.modules["tensorial.config"] = tencfg_mock
    sys.modules["tensorial.mlip.calculators.ase"] = mock.MagicMock()

    # --- Mock chemtrain.deploy (exporter needs a real base class) ---
    exporter_module = types.ModuleType("chemtrain.deploy.exporter")
    exporter_module.Exporter = _FakeBaseExporter
    deploy_module = types.ModuleType("chemtrain.deploy")
    deploy_module.exporter = exporter_module
    deploy_module.graphs = mock.MagicMock()
    chemtrain_module = types.ModuleType("chemtrain")
    chemtrain_module.deploy = deploy_module
    sys.modules["chemtrain"] = chemtrain_module
    sys.modules["chemtrain.deploy"] = deploy_module
    sys.modules["chemtrain.deploy.exporter"] = exporter_module

    # --- Mock ase ---
    sys.modules["ase"] = mock.MagicMock()

    return {
        "mock_model": mock_model,
        "mock_result": mock_result,
        "tencfg": tencfg_mock,
        "ckpt_loader": ckpt_loader,
        "ckpt": mock_ckpt,
    }


def _load_module():
    """Mock heavy deps, import export-model.py, return (module, mocks)."""
    mocks = _setup_mocks()
    mod = importlib.util.module_from_spec(_SPEC)
    sys.modules["export_model"] = mod
    _SPEC.loader.exec_module(mod)
    return mod, mocks


def _write_cfg(tmp_path, cfg):
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w") as f:
        omegaconf.OmegaConf.save(cfg, f)
    return cfg_file


def _load_cfg(minimal_config):
    return omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.load(minimal_config),
    )


def _nequip_layers(n):
    return [
        {"_target_": "tensorial.mlip.calculators.ase.SomeNequipLayer"}
        for _ in range(n)
    ]


def _post_layers():
    return [
        {"_target_": "tensorial.deploy.exporter.NodewiseReduce"},
        {"_target_": "tensorial.deploy.exporter.Grad"},
        {"_target_": "tensorial.deploy.exporter.CalcStresses"},
    ]


# ---------------------------------------------------------------------------
# Unit tests for parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:

    def test_parse_args_returns_paths_and_unit(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        ckpt_file = tmp_path / "ckpt.pt"
        ckpt_file.write_text("")

        mod, _ = _load_module()

        monkeypatch.setattr(
            sys, "argv",
            ["export-model.py", "-c", str(config_file),
             "-ck", str(ckpt_file), "-u", "lj"],
        )
        args = mod.parse_args()

        assert args.config == config_file
        assert args.ckpt == ckpt_file
        assert args.unit == "lj"

    def test_long_option_names(self, tmp_path, monkeypatch):
        mod, _ = _load_module()

        monkeypatch.setattr(
            sys, "argv",
            ["export-model.py", "--config", "c.yaml",
             "--ckpt", "k.pt", "--unit", "real"],
        )
        args = mod.parse_args()

        assert args.config == pathlib.Path("c.yaml")
        assert args.ckpt == pathlib.Path("k.pt")
        assert args.unit == "real"

    def test_defaults(self, monkeypatch):
        mod, _ = _load_module()

        monkeypatch.setattr(
            sys, "argv", ["export-model.py", "-c", "c.yaml", "-ck", "k.pt"],
        )
        args = mod.parse_args()

        assert args.unit == "metal"
        assert args.layers == _DEFAULT_LAYERS_TO_STRIP

    @pytest.mark.parametrize(
        "unit_input, expected",
        [
            ("lj", "lj"),
            ("'real'", "real"),
            ('"metal"', "metal"),
            ("  real  ", "real"),
        ],
    )
    def test_unit_strip_quotes(self, monkeypatch, unit_input, expected):
        mod, _ = _load_module()

        monkeypatch.setattr(
            sys, "argv",
            ["export-model.py", "-c", "c.yaml", "-ck", "k.pt",
             "-u", unit_input],
        )
        args = mod.parse_args()
        assert args.unit == expected

    def test_layers_is_parsed_as_int(self, monkeypatch):
        mod, _ = _load_module()

        monkeypatch.setattr(
            sys, "argv",
            ["export-model.py", "-c", "c.yaml", "-ck", "k.pt", "-l", "2"],
        )
        args = mod.parse_args()
        assert args.layers == 2


# ---------------------------------------------------------------------------
# Unit tests for TensorialExporter
# ---------------------------------------------------------------------------


class TestTensorialExporter:

    @staticmethod
    def _make(tmp_path, cfg, unit_style="real",
              layers_to_strip=_DEFAULT_LAYERS_TO_STRIP):
        """Write cfg, load the module, and build a TensorialExporter."""
        cfg_file = _write_cfg(tmp_path, cfg)
        mod, mocks = _load_module()
        exporter = mod.TensorialExporter(
            str(cfg_file), "dummy_ckpt", unit_style, layers_to_strip,
        )
        return exporter, mocks, mod

    def test_init_stores_unit_style(self, tmp_path, minimal_config):
        cfg = _load_cfg(minimal_config)
        exporter, _, _ = self._make(tmp_path, cfg, unit_style="lj")
        assert exporter.unit_style == "lj"

    def test_init_sets_r_cutoff(self, tmp_path, minimal_config):
        cfg = _load_cfg(minimal_config)
        cfg["r_max"] = 5.5
        exporter, _, _ = self._make(tmp_path, cfg)
        assert exporter.r_cutoff == 5.5

    def test_init_sets_n_layers_from_nequip_layers(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        cfg["model"]["model"]["layers"] = _nequip_layers(2) + _post_layers()
        exporter, _, _ = self._make(tmp_path, cfg)
        # Count of Nequip layers determines nbr_order
        assert exporter.nbr_order == [2, 2]

    def test_nbr_order_counted_before_stripping(
        self, tmp_path, minimal_config,
    ):
        """nbr_order must not depend on how many layers are stripped."""
        cfg = _load_cfg(minimal_config)
        cfg["model"]["model"]["layers"] = _nequip_layers(2) + _post_layers()
        exporter, _, _ = self._make(tmp_path, cfg, layers_to_strip=0)
        assert exporter.nbr_order == [2, 2]

    def test_init_sets_atomic_numbers(self, tmp_path, minimal_config):
        cfg = _load_cfg(minimal_config)
        cfg["from_data"]["atomic_numbers"] = [1, 6, 8, 16]
        exporter, _, _ = self._make(tmp_path, cfg)
        assert list(exporter.atomic_numbers) == [1, 6, 8, 16]

    @pytest.mark.parametrize(
        "layers_to_strip, expected_len",
        [(0, 7), (1, 6), (2, 5), (3, 4)],
    )
    def test_init_strips_layers(
        self, tmp_path, minimal_config, layers_to_strip, expected_len,
    ):
        cfg = _load_cfg(minimal_config)
        # 4 Nequip layers + 3 post-processing layers = 7 total
        cfg["model"]["model"]["layers"] = _nequip_layers(4) + _post_layers()
        _, mocks, _ = self._make(
            tmp_path, cfg, layers_to_strip=layers_to_strip,
        )

        # tencfg.instantiate receives cfg.model.model with stripped layers
        model_cfg = mocks["tencfg"].instantiate.call_args[0][0]
        assert len(model_cfg["layers"]) == expected_len

    def test_init_strip_three_keeps_only_nequip_layers(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        cfg["model"]["model"]["layers"] = _nequip_layers(4) + _post_layers()
        _, mocks, _ = self._make(tmp_path, cfg, layers_to_strip=3)

        model_cfg = mocks["tencfg"].instantiate.call_args[0][0]
        assert all(
            layer["_target_"].endswith("NequipLayer")
            for layer in model_cfg["layers"]
        )

    def test_init_uses_instantiated_model(self, tmp_path, minimal_config):
        cfg = _load_cfg(minimal_config)
        exporter, mocks, _ = self._make(tmp_path, cfg)
        assert exporter.model is mocks["mock_model"]

    def test_init_loads_checkpoint_params(self, tmp_path, minimal_config):
        cfg = _load_cfg(minimal_config)
        exporter, mocks, _ = self._make(tmp_path, cfg)

        mocks["ckpt_loader"].assert_called_once_with("dummy_ckpt")
        assert exporter.params["layer_0"]["weight"].shape == (10, 10)

    def test_energy_fn_returns_correct_shape(
        self, tmp_path, minimal_config, minimal_graph, particle_data,
    ):
        cfg = _load_cfg(minimal_config)
        exporter, mocks, _ = self._make(tmp_path, cfg)
        mocks["mock_result"].nodes.__getitem__.return_value = jnp.zeros((4, 1))

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
        cfg = _load_cfg(minimal_config)
        cfg["from_data"]["atomic_numbers"] = [1, 6, 8]
        exporter, mocks, _ = self._make(tmp_path, cfg)

        # Non-zero per-atom energies so masking is observable
        mocks["mock_result"].nodes.__getitem__.return_value = jnp.array(
            [[1.0], [2.0], [3.0], [4.0]], dtype=jnp.float32,
        )

        positions = minimal_graph.nodes["positions"]
        particle_data = {
            "species": jnp.array([0, 1, 2, 0], dtype=jnp.int32),
            "valid_mask": jnp.array([True, True, False, True]),
        }
        graph = jraph.GraphsTuple(
            nodes={"positions": positions,
                   "atomic_numbers": jnp.array([[1], [6], [8], [1]])},
            edges={},
            globals={},
            senders=jnp.array([0, 1, 2], dtype=jnp.int32),
            receivers=jnp.array([1, 2, 3], dtype=jnp.int32),
            n_node=jnp.array([4]),
            n_edge=jnp.array([3]),
        )

        energy = exporter.energy_fn(positions, particle_data, graph)
        # Invalid particle (index 2) is zeroed, the others are untouched
        assert jnp.allclose(energy, jnp.array([1.0, 2.0, 0.0, 4.0]))

    def test_energy_fn_builds_graph_for_model(
        self, tmp_path, minimal_config, minimal_graph,
    ):
        """species are mapped to atomic numbers; graph metadata is passed."""
        cfg = _load_cfg(minimal_config)
        cfg["from_data"]["atomic_numbers"] = [1, 6, 8]
        exporter, mocks, _ = self._make(tmp_path, cfg)
        mocks["mock_result"].nodes.__getitem__.return_value = jnp.zeros((4, 1))

        positions = minimal_graph.nodes["positions"]
        particle_data = {
            "species": jnp.array([0, 1, 2, 0], dtype=jnp.int32),
            "valid_mask": jnp.ones(4, dtype=bool),
        }
        graph = jraph.GraphsTuple(
            nodes={}, edges={}, globals={},
            senders=jnp.array([0, 1, 2], dtype=jnp.int32),
            receivers=jnp.array([1, 2, 3], dtype=jnp.int32),
            n_node=jnp.array([4]),
            n_edge=jnp.array([3]),
        )

        exporter.energy_fn(positions, particle_data, graph)

        params_arg, g = mocks["mock_model"].apply.call_args[0]
        assert params_arg is exporter.params
        assert jnp.array_equal(
            g.nodes["atomic_numbers"], jnp.array([[1], [6], [8], [1]]),
        )
        assert jnp.array_equal(g.nodes["positions"], positions)
        assert jnp.array_equal(g.senders, graph.senders)
        assert jnp.array_equal(g.receivers, graph.receivers)
        assert list(g.n_node) == [4]
        assert list(g.n_edge) == [3]

    def test_energy_fn_unwraps_irreps_array(
        self, tmp_path, minimal_config, minimal_graph, particle_data,
    ):
        """Outputs exposing `.array` (IrrepsArray) are unwrapped."""
        cfg = _load_cfg(minimal_config)
        exporter, mocks, _ = self._make(tmp_path, cfg)

        wrapped = types.SimpleNamespace(
            array=jnp.array([[1.0], [2.0], [3.0], [4.0]], dtype=jnp.float32),
        )
        mocks["mock_result"].nodes.__getitem__.return_value = wrapped

        energy = exporter.energy_fn(
            minimal_graph.nodes["positions"],
            {**particle_data, "valid_mask": jnp.ones(4, dtype=bool)},
            minimal_graph,
        )
        assert energy.shape == (4,)
        assert jnp.allclose(energy, jnp.array([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# Unit tests for export_model function
# ---------------------------------------------------------------------------


class TestExportModelFunction:

    @staticmethod
    def _mock_export_model_call(tmp_path, cfg):
        """Load the module with TensorialExporter replaced by a mock.

        Returns (mod, cfg_file, ckpt_file, exporter_mock).
        """
        cfg_file = _write_cfg(tmp_path, cfg)
        ckpt_file = tmp_path / "ckpt.pt"
        ckpt_file.write_text("")

        mod, _ = _load_module()

        exporter_mock = mock.MagicMock()
        exporter_mock.export = mock.MagicMock()
        exporter_mock.save = mock.MagicMock()

        # Replace the class in the module namespace so `TensorialExporter(...)`
        # returns our mock instance.
        mod.TensorialExporter = mock.MagicMock(return_value=exporter_mock)

        return mod, cfg_file, ckpt_file, exporter_mock

    def test_export_model_creates_output_directory(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        output_dir = tmp_path / "nested" / "output"

        mod, cfg_file, ckpt_file, _ = self._mock_export_model_call(
            tmp_path, cfg,
        )
        mod.export_model(
            output_dir, cfg_file, ckpt_file, "lj", _DEFAULT_LAYERS_TO_STRIP,
        )

        assert output_dir.is_dir()

    def test_export_model_existing_output_directory_ok(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mod, cfg_file, ckpt_file, exporter_mock = self._mock_export_model_call(
            tmp_path, cfg,
        )
        mod.export_model(
            output_dir, cfg_file, ckpt_file, "lj", _DEFAULT_LAYERS_TO_STRIP,
        )

        exporter_mock.save.assert_called_once()

    def test_export_model_builds_exporter_with_all_args(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, _ = self._mock_export_model_call(
            tmp_path, cfg,
        )
        mod.export_model(output_dir, cfg_file, ckpt_file, "lj", 2)

        mod.TensorialExporter.assert_called_once_with(
            cfg_file, ckpt_file, "lj", 2,
        )

    def test_export_model_exports_cpu_and_cuda(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, exporter_mock = self._mock_export_model_call(
            tmp_path, cfg,
        )
        mod.export_model(
            output_dir, cfg_file, ckpt_file, "lj", _DEFAULT_LAYERS_TO_STRIP,
        )

        exporter_mock.export.assert_called_once_with(
            platforms=("cpu", "cuda"),
        )

    def test_export_model_saves_protobuf_file(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, exporter_mock = self._mock_export_model_call(
            tmp_path, cfg,
        )
        mod.export_model(
            output_dir, cfg_file, ckpt_file, "lj", _DEFAULT_LAYERS_TO_STRIP,
        )

        exporter_mock.save.assert_called_once_with(
            output_dir / "tensorial-lj.ptb",
        )

    def test_export_model_exports_before_saving(
        self, tmp_path, minimal_config,
    ):
        cfg = _load_cfg(minimal_config)
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, exporter_mock = self._mock_export_model_call(
            tmp_path, cfg,
        )
        mod.export_model(
            output_dir, cfg_file, ckpt_file, "lj", _DEFAULT_LAYERS_TO_STRIP,
        )

        call_names = [c[0] for c in exporter_mock.method_calls]
        assert call_names.index("export") < call_names.index("save")

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
        cfg = _load_cfg(minimal_config)
        output_dir = tmp_path / "output"

        mod, cfg_file, ckpt_file, exporter_mock = self._mock_export_model_call(
            tmp_path, cfg,
        )
        mod.export_model(
            output_dir, cfg_file, ckpt_file, unit_style,
            _DEFAULT_LAYERS_TO_STRIP,
        )

        save_path = exporter_mock.save.call_args[0][0]
        assert save_path == output_dir / expected_filename


# ---------------------------------------------------------------------------
# Unit tests for main()
# ---------------------------------------------------------------------------


class TestMain:

    @staticmethod
    def _run_main(tmp_path, monkeypatch, minimal_config, extra_argv=()):
        cfg_file = _write_cfg(tmp_path, _load_cfg(minimal_config))
        ckpt_file = tmp_path / "ckpt.pt"
        ckpt_file.write_text("")

        mod, _ = _load_module()

        exporter_mock = mock.MagicMock()
        mod.TensorialExporter = mock.MagicMock(return_value=exporter_mock)

        monkeypatch.setattr(sys, "argv", [
            "export-model.py",
            "-c", str(cfg_file),
            "-ck", str(ckpt_file),
            *extra_argv,
        ])
        monkeypatch.chdir(tmp_path)

        mod.main()
        return mod, cfg_file, ckpt_file, exporter_mock

    def test_main_calls_export_model_with_cwd(
        self, tmp_path, monkeypatch, minimal_config,
    ):
        _, _, _, exporter_mock = self._run_main(
            tmp_path, monkeypatch, minimal_config, ["-u", "lj"],
        )

        # Output goes to the current working directory
        exporter_mock.save.assert_called_once_with(
            pathlib.Path.cwd() / "tensorial-lj.ptb",
        )

    def test_main_forwards_cli_arguments(
        self, tmp_path, monkeypatch, minimal_config,
    ):
        mod, cfg_file, ckpt_file, _ = self._run_main(
            tmp_path, monkeypatch, minimal_config, ["-u", "real"],
        )

        # Default for --layers is used when not given on the command line
        mod.TensorialExporter.assert_called_once_with(
            cfg_file, ckpt_file, "real", _DEFAULT_LAYERS_TO_STRIP,
        )

    def test_main_default_unit_is_metal(
        self, tmp_path, monkeypatch, minimal_config,
    ):
        _, _, _, exporter_mock = self._run_main(
            tmp_path, monkeypatch, minimal_config,
        )

        exporter_mock.save.assert_called_once_with(
            pathlib.Path.cwd() / "tensorial-metal.ptb",
        )
