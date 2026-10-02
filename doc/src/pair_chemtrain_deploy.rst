.. index:: pair_style chemtrain
.. index:: pair_style chemtrain/kk

pair_style chemtrain command
============================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style chemtrain backend [memory_fraction] keyword values ...
   pair_coeff * * model.ptb ...

* ``backend`` = ``cpu`` or ``cuda``
* ``memory_fraction`` = optional CUDA memory fraction, accepted but ignored for CPU
* zero or more keyword/value groups may follow

.. parsed-literal::

   comm *value*
   capacity/atom *factor*
   capacity/edge *factor*
   capacity/dense-neighbor *factor*
   models *name scale* ...
   atom/input [*model_context*] *field source*
   global/input [*model_context*] *field source*
   output [*model_context*] *output_name*
   topology

Examples
""""""""

.. code-block:: LAMMPS

   units lj
   pair_style chemtrain cuda
   pair_coeff * * lennard_jones_sparse.ptb

Description
"""""""""""

The ``chemtrain`` style evaluates models exported by chemtrain-deploy and adds
their energies, forces, and virials to LAMMPS. Atom types are passed as
zero-based species. Model files use the ``.ptb`` suffix and contain metadata
for the neighbor list, unit style, inputs, outputs, and supported backends.

Backend and runtime
"""""""""""""""""""

The standard ``chemtrain`` pair style supports the ``cpu`` backend, including
models that use host communication callbacks. The ``cuda`` backend is
currently the only LAMMPS configuration covered by automated end-to-end tests.
The CPU host pair style has also been validated manually with single-rank and
MPI runs. A model bundle can contain CPU and CUDA implementations, and the
connector selects one that matches the requested backend.

Format 5 bundles contain separate executables for comm off with Newton pair
off and comm off with Newton pair on. Communication-capable bundles also
contain comm on with Newton pair on. LAMMPS selects the exact communication
and Newton combination and does not fall back to a differently configured
executable. Older bundles must be re-exported.

The optional CUDA ``memory_fraction`` defaults to 0.75 and is ignored by the
CPU backend. GPU selection is controlled by the PJRT runtime, MPI rank mapping,
and Kokkos command-line options. The removed ``device`` keyword is not
accepted.

Model loading and neighbors
"""""""""""""""""""""""""""

Without named model contexts, ``pair_coeff * *`` takes exactly one model file.
Coefficients must cover all atom types. Sparse and dense host neighbor
representations are supported. Kokkos device execution currently requires a
``SIMPLE_SPARSE`` model.

Model contexts
""""""""""""""

The ``models`` keyword evaluates multiple exported models and scales their
contributions. The keyword must appear before any model-specific input or
output keyword. Supply one ``name scale`` pair per model, then provide model
files to ``pair_coeff`` in the same order:

.. code-block:: LAMMPS

   pair_style chemtrain cuda models short 1.0 long 0.5 \
      output short uncertainty output long uncertainty
   pair_coeff * * short.ptb long.ptb

For named contexts, ``atom/input`` and ``global/input`` take the context name
before the field and source. ``output`` and :doc:`compute_chemtrain_output`
must use the same context name.

Model inputs
""""""""""""

``atom/input`` maps a declared model particle field to a scalar custom integer
atom property. ``global/input`` maps a declared global field to a numeric
literal or an equal-style variable.

Auxiliary outputs
"""""""""""""""""

``output`` requests an auxiliary model output. The
:doc:`compute_chemtrain_output` command exposes the cached value after the
current force evaluation. ``PARTICLE`` values are per atom. ``LOCAL`` values
are additive rank-local contributions that the compute reduces once.
``GLOBAL`` values are complete configuration values that LAMMPS does not
reduce. For example, an interface-pinning bias computed after ``comm.reduce``
is ``GLOBAL`` because the bias is already complete on every rank. Force,
energy, and virial are built-in outputs and cannot be requested as auxiliary
outputs. Configuration outputs also declare whether they are extensive. This
controls LAMMPS normalization and is independent of the ``LOCAL`` or ``GLOBAL``
reduction scope.

Capacity management
"""""""""""""""""""

The connector uses fixed-capacity buffers. ``capacity/atom``,
``capacity/edge``, and ``capacity/dense-neighbor`` set multiplicative growth
factors for retries after an overflow. Every factor must be greater than one.
Capacity growth and retry are collective for MPI execution.

Communication and MPI
"""""""""""""""""""""

``comm on`` selects the communication-enabled Newton-on model variant and
registers the callbacks called by the exported executable.
Communication-enabled models require ``newton pair on``. The standard pair
style supports host callbacks on CPU and CUDA. Kokkos device communication
strictly requires CUDA-aware MPI; there is no ``/kk/host`` or host-staged
Kokkos fallback.

Bundles that call a global communication reduction require communication.
LAMMPS rejects ``comm off`` for those bundles, even on a single MPI rank.
Models that only need neighbor-list rows for ghost atoms remain valid with
``comm off``.

Virial
""""""

Format 5 model bundles always compute and return
:math:`V=-dU_\mathrm{local}/d(e_{xx},e_{yy},e_{zz},e_{xy},e_{xz},e_{yz})`.
The exporter applies the lower-triangular deformation
:math:`r'=((1+e_{xx})x,e_{xy}x+(1+e_{yy})y,e_{xz}x+e_{yz}y+(1+e_{zz})z)`
while graph connectivity remains fixed. The result has the model's energy
units and uses the LAMMPS component order ``xx, yy, zz, xy, xz, yz``. The
strain derivative supports both Newton settings and does not reconstruct the
virial from accumulated atom forces. LAMMPS binds the six-component output on
every model evaluation and adds it to the global virial only when the current
calculation requests one. Per-atom stress is not provided.

Topology
""""""""

Models may request aligned 1-2, 1-3, 1-4, and 1-5 topology categories.
``topology`` installs zero-valued bond, angle, dihedral, and improper styles
when a topology is present but no corresponding force style was defined.
Categories follow LAMMPS special-neighbor construction, including angle and
dihedral trimming.

Accelerator variants
""""""""""""""""""""

The KOKKOS package provides ``chemtrain/kk`` and ``chemtrain/kk/device``. The
variants require the CUDA backend. Use the standard ``chemtrain`` style for
CPU execution. Dense-neighbor models are not supported by Kokkos execution.
Multi-rank ``chemtrain/kk/device`` communication strictly requires CUDA-aware
MPI. ``chemtrain/kk/host`` is not supported.

----------

.. include:: accel_styles.rst

----------

Restrictions
""""""""""""

The pair style is available only when LAMMPS is built with the
:ref:`CHEMTRAIN-DEPLOY package <PKG-CHEMTRAIN-DEPLOY>`. Build instructions and
the external connector-library contract are described in
:ref:`chemtrain-deploy-build`.

Citation
""""""""

If you use the ``chemtrain`` pair style, please cite the chemtrain-deploy and
chemtrain publications: :ref:`(Fuchs2025Deploy) <Fuchs2025Deploy>` and
:ref:`(Fuchs2025Chemtrain) <Fuchs2025Chemtrain>`.

Related commands
""""""""""""""""

:doc:`compute_chemtrain_output`, :doc:`pair_coeff`, :doc:`newton`

----------

.. _Fuchs2025Deploy:

**(Fuchs2025Deploy)** P. Fuchs, W. Chen, S. Thaler, and J. Zavadlav,
chemtrain-deploy: A Parallel and Scalable Framework for Machine Learning
Potentials in Million-Atom MD Simulations, Journal of Chemical Theory and
Computation, 21, 7550-7560 (2025),
`DOI: 10.1021/acs.jctc.5c00996 <https://doi.org/10.1021/acs.jctc.5c00996>`_.

.. _Fuchs2025Chemtrain:

**(Fuchs2025Chemtrain)** P. Fuchs, S. Thaler, S. Roecken, and
J. Zavadlav, chemtrain: Learning Deep Potential Models via Automatic
Differentiation and Statistical Physics, Computer Physics Communications,
310, 109512 (2025),
`DOI: 10.1016/j.cpc.2025.109512 <https://doi.org/10.1016/j.cpc.2025.109512>`_.
