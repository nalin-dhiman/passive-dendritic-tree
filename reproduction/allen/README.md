# Reconstructed cortical-arbor comparison

This implementation rebuilds a fixed passive electrical model directly from the
Allen Cell Types Database morphology of specimen **588402092**. It independently
selects the largest basal primary subtree and seven physical ports using geometry
alone, then executes the complete spatial/frequency comparison.

From the repository root:

```sh
python reproduction/allen/run.py --output results/allen --include-transients
```

The output directory must be new or empty. To verify an existing output:

```sh
python reproduction/allen/run.py --output results/allen --include-transients --verify-only
```

## What runs

- Four configurations: baseline, an unchanged-copy control, uniform dendritic
  radius factor 0.97, and selective thinning of the largest basal primary subtree.
- Four nested meshes: 1, 2, 4 and 8 subdivisions per original edge.
- Eight frequencies: 0, 1, 2, 5, 10, 20, 50 and 100 Hz.
- A sparse-matrix solution and a separately implemented tree-elimination solution,
  with current balance, reciprocity, passivity, mesh and material checks.
- 128 harmonic conditions, including the unchanged-copy control.
- Primary full-matrix-normalized and secondary row-normalized electrical metrics,
  identified in separate CSV files.
- Complex current patterns with zero root voltage difference, steady-state port
  responses, and directly solved full-node harmonic fields.
- With `--include-transients`, all 18 rectangular-current executions: three
  configurations, somatic/apical input, and timesteps 0.1, 0.05 and 0.025 ms.

The nominal passive parameters are axial resistivity 100 ohm cm, membrane
conductance 0.0001 S/cm2 and capacitance 1 uF/cm2. The soma is a fixed sphere.
Soma-to-stem edges use child-radius cylinders; internal edges are linear-radius
frusta, with exact half-frustum control-volume membrane areas. Axon is excluded.

Uniform and selective configurations each remove 5.91% of modeled dendritic
volume. Soma-plus-dendrite volume decreases by approximately 2.6945%, because
the soma stays fixed. Coordinates, radius units, source IDs and exact source
hash are recorded in `data/SOURCE.json`.

## Accuracy and expected failures

The harmonic numerical checks pass for this specified finite-port experiment.
The electrical-preservation criteria can fail and are reported separately from
numerical accuracy. In particular, stronger distributed criteria reject the
material configurations even when the somatic criterion accepts them.

The distal rectangular-current comparison **fails** its fixed 0.0025 mV
finest-pair temporal-accuracy requirement. Running with `--include-transients`
reproduces that failure. The process exits zero only when the numerical outputs
agree with the saved reference results, including this expected failed accuracy
criterion. `summary.json` and `reproduction_verification.json` distinguish
faithful reproduction from scientific gate judgments.

The implemented engineering metric allowance is `abs(metric8 - metric4)`.
A separate sensitivity table also reports
`abs(metric8 - metric4) + abs(metric4 - metric2)`. Neither is a rigorous continuum
error bound or anatomical confidence interval. The 5% electrical margin and
0.0025 mV transient requirement are unchanged.

## Source and reuse

Morphology: Allen Institute for Brain Science, Allen Cell Types Database,
[specimen 588402092](https://celltypes.brain-map.org/experiment/morphology/588402092),
[exact source SWC](https://api.brain-map.org/api/v2/well_known_file_download/675965556).
The original SWC is included unchanged. Its header retains the source tracing
provenance as supplied by Allen.

Allen-derived data are provided for attributed noncommercial educational/research
use under the [Terms of Use](https://alleninstitute.org/legal/terms-of-use) and
[Citation Policy](https://alleninstitute.org/legal/citation-policy).
See the repository `NOTICES.md` for code/data rights. No completeness or radius
calibration is inferred from a successful electrical calculation.

These calculations describe the retained passive geometry, seven collocated
ports and sampled frequencies. They contain no active channels, optimization,
neuronal task, or metabolic accounting.
