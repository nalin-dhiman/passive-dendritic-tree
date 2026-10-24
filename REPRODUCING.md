# Reproduce the passive electrical experiments

The browser is an interactive workbench. The commands below regenerate the
complete deterministic Y-cable and reconstructed-arbor comparisons from their
geometry and fixed parameter files, independently of the browser animation.

## One complete run

Use Python 3.10–3.12 (tested with Python 3.12.2) and Node.js 18 or later for
the browser fixtures. From a new checkout:

```sh
git clone https://github.com/nalin-dhiman/passive-dendritic-tree.git
cd passive-dendritic-tree
python3 -m venv .venv
```

Activate the environment on Linux/macOS with `source .venv/bin/activate`, or
on Windows PowerShell with `.venv\Scripts\Activate.ps1`. Then run:

```sh
python -m pip install -r requirements-reproduce.lock
npm test
python -m unittest discover -s tests -p "test_*.py" -v
python reproduce.py --output outputs/full
```

Choose a new output path for each run. The programs refuse to overwrite previous
results. No NEURON installation, GPU, TeX installation, external anatomical
download, decoder training or optimization is needed.

The lock records the tested scientific and plotting packages. It is not an
operating-system image or a guarantee of bitwise equivalence across BLAS
implementations. Numerical comparison tolerances and execution metadata make
the intended checks explicit.

## What runs

| Experiment | Fixed comparison | Independent checks |
|---|---|---|
| Ideal Y cable | Five configurations × seven parameter settings × 101 frequencies, with seven fixed ports | Two continuum formulations, finite-volume meshes 16/48/144, material identities, reciprocity, passivity, current conservation and constructed current witnesses |
| Allen cortical arbor | Four configurations × four spatial meshes × eight frequencies | Sparse matrix solve versus tree elimination, spatial contraction, geometry invariants, full-matrix and secondary row-normalized comparisons |
| Reconstructed rectangular-current responses | Three configurations × two input sites × three timesteps = 18 integrations | Matched physical recordings, common-time voltage comparisons and the unchanged temporal accuracy criterion |

The unchanged and disruptive controls are retained. Numerical reference files
contain saved arrays and expected values; the runners first calculate new
solutions and then compare them with those references. Replaying a reference
array alone is not treated as reproduction.

## Expected findings and the retained failure

Uniform and selective radius changes each remove 5.91% of modeled dendritic
volume. Their electrical responses differ. The Y comparison preserves its
row-normalized gain definitions; the reconstructed comparison preserves its
primary full-matrix normalization and separately reports the secondary
row-normalized calculation. The browser's Y port ordering and gain normalization
are explicitly different from the canonical research-Y ordering and row gains;
do not compare their displayed percentages without accounting for these choices.

The full reconstructed rectangular-current family **fails** its declared
0.0025 mV finest-pair temporal accuracy requirement for distal-apical input.
Reproducing that failure is an expected successful reproduction of the recorded
experiment. It is never converted into a scientific PASS. Harmonic acceptance,
temporal acceptance and reference agreement are saved separately.

The reconstructed metric allowance uses the historical implemented absolute
change between mesh 8 and mesh 4. The sum of two successive changes is a
separately labeled sensitivity calculation. These are engineering discrepancy
indicators, not biological confidence intervals or rigorous continuum bounds.

## Files produced

`outputs/full/y_cable/` contains newly calculated impedance matrices,
finite-volume comparisons, material/criterion tables and current witnesses.
`outputs/full/allen/` contains newly calculated geometry, matrices, convergence
tables, seven-port harmonic responses and all requested transient traces.
`outputs/full/figures/` contains vector PDF/PNG graphics derived from those
new results and a figure-source manifest.

`REPRODUCTION.json` records command exit codes, runtime and output hashes.
Experiment-specific summaries distinguish numerical gates, reference agreement
and electrical-preservation verdicts. Individual stdout/stderr files remain
available if a command fails.

For a partial frequency-domain run:

```sh
python reproduce.py --output outputs/harmonic_only --harmonic-only
```

That run is explicitly labeled partial and does not reproduce the transient
family. Individual runners also accept `--help`.

## Geometry, source use and limits

The cortical example uses Allen Cell Types specimen 588402092. Exact source
identifiers, SWC SHA-256, units, retained compartments and source-use terms are
included with the anatomical input. The original soma remains fixed and the
axon is excluded. The two cable implementations preserve physical source-node
identities during refinement.

The experiments concern specified passive geometries, seven-port input/output
spaces and sampled frequencies. Radius uncertainty, missing arbors and omitted
membrane structures remain anatomical limitations. Material and electrical
comparisons are not biological ATP measurements. Allen-derived material remains
subject to the noncommercial-use and attribution terms in [NOTICES.md](NOTICES.md).
