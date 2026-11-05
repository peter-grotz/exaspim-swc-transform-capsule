# exaspim-swc-transform-capsule

Transforms exaSPIM SWC reconstructions from specimen space into CCF space, using
[aind_exaspim_register_cells](https://github.com/AllenNeuralDynamics/aind_exaspim_register_cells).

## Usage

Reads the specimen-space SWCs in a `refinement/final-world/` directory and the sample's
registration, and writes CCF-space SWCs to `/results/alignment/aligned_swcs` with a
`data_process.json` beside them.

Parameters, in App Builder order:

1. `swc_dir`: the `final-world/` directory.
2. `processed_dataset`: optional; resolved from the reconstructions' record if empty.
3. `experimenters`: comma-separated.
4. `allow_no_displacement_field`: the run fails when the sample has no manual CCF
   refinement in `<dataset>/manual_ccf_refinement/`, unless this is `True`.

A reconstruction that fails to transform is logged and skipped, and the capsule exits
non-zero. Pass `--fail-fast` to abort on the first failure instead.

**Python 3.11.** `allensdk` pins `numpy<1.24`, which cannot build on 3.12.

## Level of Support

![support](https://img.shields.io/badge/support-supported-brightgreen)

## Installation

Metadata and naming come from
[exaspim-swc-processing](https://github.com/peter-grotz/exaspim-swc-processing); the ANTs
stack stays here because of its pinned numeric dependencies.
