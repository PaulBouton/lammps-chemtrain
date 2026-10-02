.. index:: compute chemtrain/output
.. index:: compute chemtrain/output/kk

compute chemtrain/output command
================================

Accelerator Variants: *chemtrain/output/kk*, *chemtrain/output/kk/device*

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID chemtrain/output [model_context] output_name

Examples
""""""""

.. code-block:: LAMMPS

   pair_style chemtrain cuda output uncertainty
   pair_coeff * * uncertainty.ptb
   compute uncertainty all chemtrain/output uncertainty

Description
"""""""""""

The ``chemtrain/output`` compute exposes an auxiliary model output requested
with the ``output`` keyword of :doc:`pair_chemtrain_deploy`. The output scope
determines how LAMMPS exposes the value.

* ``PARTICLE`` is a per-atom vector for one component or a per-atom array for
  multiple components. Values are zero for atoms outside the compute group.
* ``LOCAL`` is an additive rank-local value. The compute applies one MPI sum
  and exposes a global scalar or vector.
* ``GLOBAL`` is a complete configuration value. The compute exposes the value
  directly as a global scalar or vector without a LAMMPS reduction.

Configuration outputs (``LOCAL`` and ``GLOBAL``) require group ``all``. The
compute exposes a multi-dimensional configuration output as a one-dimensional
vector while preserving the component order declared by the exported model.
The exporter also marks each configuration output as intensive or extensive.
LAMMPS uses that metadata when normalizing thermodynamic output. It does not
change whether the compute performs an MPI reduction.

Output lifetime
"""""""""""""""

Values correspond to the most recent chemtrain force evaluation. Defining the
compute before ``pair_coeff`` or reading its values before a force evaluation
is an error. The compute does not evaluate the model independently.

Model contexts
""""""""""""""

For one unnamed model, omit ``model_context``. When the pair style declares
contexts with ``models``, ``model_context`` is required and must match the
context used by the pair style's ``output`` keyword:

.. code-block:: LAMMPS

   pair_style chemtrain cuda models short 1.0 long 0.5 output short uncertainty
   pair_coeff * * short.ptb long.ptb
   compute uncertainty all chemtrain/output short uncertainty

Accelerator variants
""""""""""""""""""""

The KOKKOS variants are ``chemtrain/output/kk`` and
``chemtrain/output/kk/device``. Both variants expose the same three output
scopes. The compute variant must match the active pair-style execution path.

----------

.. include:: accel_styles.rst

----------

Restrictions
""""""""""""

The compute requires an active ``chemtrain`` pair style and the
:ref:`CHEMTRAIN-DEPLOY package <PKG-CHEMTRAIN-DEPLOY>`.
