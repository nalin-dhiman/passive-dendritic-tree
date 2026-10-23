# Passive Cable Lab

A buildless, browser-local demonstration of voltage spread through a passive cable tree. Change cable radii, apply an electrode-current pulse, compare root and distal traces, and export the numerical samples. A local SWC upload lets you explore your own geometry without sending that file to a server.

**This model has no active channels, synapses, spikes, optimization, or metabolic accounting.** A small difference at one displayed location is not evidence that every possible electrical response is preserved.

Live demonstration: <https://nalin-dhiman.github.io/passive-dendritic-tree/>

## Open the demonstration

The static site is in `docs/`. For a local preview:

```sh
python3 -m http.server 8080 --directory docs
```

Visit <http://localhost:8080>. JavaScript modules and a local Web Worker require HTTP(S); opening `index.html` as a `file:` URL is unsupported. There are no third-party browser scripts, CDN dependencies, tracking services, or remote simulation calls.

For GitHub Pages, publish the `docs/` directory from the default branch, or use a static Pages workflow that uploads that directory.

## Electrical model

All lengths and radii are in µm. Cable edges have linearly varying radii. Each graph node owns half the lateral membrane of every incident edge. A shared node enforces branch-voltage continuity, while pairwise axial currents conserve current. Terminal ends are sealed; end-cap membrane is omitted. The model is a finite-volume approximation of a passive cable tree.

For an edge of length `L` with endpoint radii `r0` and `r1`:

```text
volume = π L (r0² + r0 r1 + r1²) / 3                     [µm³]
lateral area = π (r0 + r1) √(L² + (r1 − r0)²)           [µm²]
axial resistance = 0.01 Ra L / (π r0 r1)                 [MΩ]
axial conductance = 100 π r0 r1 / (Ra L)                 [µS]
```

For a control-volume membrane area `A`:

```text
C = cm A × 10⁻⁵                                        [nF]
g_leak = gm A × 0.01                                   [µS]
C dV/dt = I − g_leak (V − E_L) − Σ g_axial (V − V_neighbor)
```

`Ra` is in Ω cm, `cm` in µF/cm², and `gm` in S/cm². Currents are in nA, voltages in mV and time in ms. The defaults are `Ra=150`, `cm=1`, `gm=0.0001`, and `E_L=−65 mV`. At rest every node starts at the leak reversal. The current pulse starts at 20 ms. The browser integrates both original and modified geometries with identical inputs.

Backward Euler with a 0.05 ms timestep is solved by linear-time Hines elimination. Pulse current is evaluated at each integration interval's midpoint. The animation displays actual numerical solutions sampled every 0.2 ms, rather than a prerecorded sequence. The plots, animation and CSV share those same samples. Spatial and temporal discretization errors remain; the default settings are not a certified accuracy guarantee for arbitrary uploaded trees.

The synthetic Y has a 300 µm parent cable of radius 2 µm and two 400 µm daughter cables of radius 1.5 µm. Mesh edges are at most 10 µm long. `All cables` scales every cable radius; `Selected branch` additionally scales daughter B. Centerlines, topology, current location and passive densities stay fixed. The material panel reports cable volume and lateral area, not an energetic cost.

## Local SWC input

SWC files must contain exactly seven whitespace-separated columns: node ID, type, x, y, z, radius, parent ID. Use µm for coordinates and **radius**, not diameter. One parent-`−1` root is required; IDs must be unique and radii strictly positive. Unsupported or malformed graphs fail explicitly rather than being repaired or truncated.

The interactive limits are 2 MB and 2,500 original points. Edges longer than 15 µm are subdivided, preserving their linear-radius geometry; a 6,000-node limit prevents excessive browser resource use. Every original cable edge and type is included. This is an electrical idealization, not an anatomical completeness or radius-calibration audit.

A single type-1 root soma is represented by a sphere with area `4πr²`. Its outgoing cables use their child radius at the attachment, avoiding an artificial cone from the soma radius. Multi-point somata are rejected because their membrane representation requires a separate explicit policy. Soma area stays fixed under the cable-radius sliders and is excluded from the displayed cable-area comparison. Soma volume is not included in the cable-volume display.

For an uploaded tree, distal B is the longest root-to-terminal path endpoint and distal A is the next-longest terminal endpoint (or the root if only one exists). `Selected branch` affects only the terminal path from distal B back to its nearest upstream fork. The diagram uses the two coordinate axes with the largest spatial extents; widths are exaggerated. The current input and recordings remain at the same graph identities when radii change.

## Run the numerical fixtures

Node.js 18 or later is sufficient; no installation is required:

```sh
npm test
```

Fixtures cover analytic geometry, radius scaling, exact frustum axial resistance, membrane units, an independently soluble RC circuit, temporal error contraction, an independent dense linear solve, per-node and whole-tree current conservation, sealed-cylinder spatial convergence, Y symmetry, pulse timing, rest preservation, final recording samples, and strict SWC validation. They establish those scoped properties; they do not certify arbitrary biological reconstructions or active-neuron models.

## Files

- `docs/js/engine.mjs`: pure geometry, SWC validation and passive finite-volume solver.
- `docs/js/simulation-worker.mjs`: numerical integration outside the UI thread.
- `docs/js/app.mjs`: controls, canvas drawing and numerical CSV export.
- `tests/engine.test.mjs`: executable numerical fixtures.

No reconstructed neuron data are bundled. For uploaded data, the user remains responsible for provenance, units, anatomical interpretation and applicable data-use terms.

## Optional Python calculations

`python/passive_tree.py` provides frustum geometry, exact axial resistance, a passive finite-volume matrix, sparse frequency-domain impedance, an independent Hines solve, and Crank–Nicolson transients. Install NumPy and SciPy in your own environment:

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Example (a user-supplied, topologically ordered SWC file):

```python
import sys
sys.path.insert(0, 'python')
from passive_tree import read_swc, build_tree

swc, parent = read_swc('your_morphology.swc')
tree = build_tree(swc, parent, subdivisions=2, Ra=150, gm=1e-4, cm=1)
root = tree.original_to_mesh[0]
distal = tree.original_to_mesh[-1]  # example source point; choose your sites explicitly
Z, full_response, currents = tree.impedance([root, distal], frequency_Hz=10)
print(Z)  # mV / nA = MΩ
```

The Python reader explicitly retains only soma and dendritic types 1, 3 and 4; it requires one root soma marker, parents before children, positive radii and nonzero edges. Disconnection after filtering is rejected. Unlike the browser upload (which includes all cable types), this is a soma/dendrite-only API. A scalar Python `radius_scale` changes **all** radii, including the soma; pass an explicit per-source-node vector with root factor 1 to hold the soma fixed. This policy must be chosen explicitly for comparisons.

Python transients return voltage **relative to rest** in mV, whereas the browser plots absolute voltage with a −65 mV leak reversal. Python uses Crank–Nicolson and the browser uses backward Euler; they are separate numerical implementations with distinct timestep policies. The synthetic tests cover analytic geometry, independently integrated axial resistance, a sealed-cylinder continuum DC solution with soma load, reciprocity, passivity, geometry preservation, solver agreement, pulse charge and invalid-parameter rejection. None of these checks establishes universal numerical accuracy, anatomical completeness, radius calibration, active physiology, or biological functional equivalence.

See `NOTICES.md` for code and data-use notices.
