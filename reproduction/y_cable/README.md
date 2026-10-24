# Passive Y-cable numerical experiment

Run from the repository root:

```sh
python reproduction/y_cable/run.py --output results/y_cable
```

The output must be new or empty. The command performs all 35 fixed conditions:
five prescribed geometries, seven engineering parameter settings and 101
frequencies from 0 to 100 Hz. It recomputes every matrix and numerical check;
the bundled reference arrays are used only to verify the new results.

The seven physical ports are ordered:

1. parent start (root),
2. branch junction,
3. first daughter tip,
4. second daughter tip,
5. parent midpoint,
6. first daughter midpoint,
7. second daughter midpoint.

This order differs from the topological drawing order used by the browser.
All cylinders are 400 micrometres long. Baseline radii are 1, 0.8 and 0.8
micrometres. Uniform thinning uses 0.97 times each radius; selective thinning
uses radii 1, 0.7108108046449492 and 0.8 micrometres. Both prescribed changes
remove 5.91% of cylindrical volume. `config.json` specifies all conditions,
criteria, physical units and discrepancy allowances.

`engine.py` assembles exact continuum nodal admittances and finite volumes;
`independent_reference.py` solves separate voltage/current edge-state equations.
The finite-volume refinements use 16, 48 and 144 intervals per cylinder.
Current balance, continuity, reciprocity, passivity, material identities,
conditioning, refinement and independent-solver agreement are checked before
an electrical-preservation comparison is accepted.

The comparison code retains row-normalized root and all-port gain errors and
the complete-response-relative error `||(Zc-Z0) inverse(Z0)||2`. These are
separate from the full-matrix-normalized somatic and distributed gain measures
used for the reconstructed-arbor example. The empirical finite-volume allowances
are numerical discrepancy indicators, not rigorous continuum error bounds.

A successful command returns exit code zero and
`reproduction_status.json: reproduction_status = PASS`. This means the
calculation and its saved reference comparisons reproduced. It does **not**
change the electrical verdicts: the two material-saving geometries fail the
complete-response-relative requirement; selective thinning also fails all-port
gain preservation. Expected scientific failures are recorded in `summary.json`
and `comparison.csv`.

The raw outputs include every continuum and finite-volume impedance matrix,
endpoint current, response-relative matrix, prescribed current witness and
periodic waveform. Root-difference-null currents are signed/phase-controlled
mathematical inputs at the seven ports. Their waveforms are harmonic steady
states, not pulse-startup simulations or synaptic activity.

The implementation is deterministic and uses no random seeds or held-out data.
`source_provenance.json` records preserved numerical-source and reference hashes.
