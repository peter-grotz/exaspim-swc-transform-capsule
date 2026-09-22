# exaspim-swc-transform-capsule

Transforms exaSPIM SWC reconstructions from specimen space into CCF space, using
[aind_exaspim_register_cells](https://github.com/AllenNeuralDynamics/aind_exaspim_register_cells).

## Usage

Reads specimen-space SWCs and the registration transforms, writes CCF-space SWCs to
`/results/alignment/aligned_swcs` with a `data_process.json` beside them.

A reconstruction that fails to transform is logged and skipped so one bad cell does not
lose the run; the count and the failed stems are recorded in the stage metadata, and the
capsule exits non-zero. Pass `--fail-fast` to abort on the first failure instead.

**No GPU.** The previous pipeline requested one, but nothing here uses CUDA and antspyx is
CPU-only.

**Python 3.11.** `allensdk` pins `numpy<1.24`, which cannot build on 3.12.

## Level of Support

![support](https://img.shields.io/badge/support-supported-brightgreen)

## Installation

Metadata and naming come from
[exaspim-swc-processing](https://github.com/peter-grotz/exaspim-swc-processing); the ANTs
stack stays here because of its pinned numeric dependencies.
