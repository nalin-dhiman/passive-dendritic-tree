# Passive Cable Lab

An interactive electrical workbench for a reconstructed mouse cortical dendritic
tree and an ideal Y-shaped cable. Apply a sinusoidal current, change where a fixed
amount of modeled dendritic material is removed, and compare voltage at the soma
and across the arbor.

**[Open the live explorer](https://nalin-dhiman.github.io/passive-dendritic-tree/)**

The main explorer calculates periodic steady-state responses locally in a Web
Worker. Its arbor colors come from the complex voltage solution at the displayed
nodes. Animation shows phase within an established periodic response. It does not
show a stimulus switching on or an action potential propagating.

## Explore

- Switch between the reconstructed cortical arbor and the ideal Y cable.
- Compare the original geometry, uniform thinning and selective subtree thinning.
- Change current input, frequency, amplitude and phase; observe multiple fixed ports.
- Inspect original, modified and difference voltage maps using shared scales.
- Change projection, pan and zoom without altering the electrical model.
- Compare material measures, frequency-dependent transfer and three explicitly
  defined electrical discrepancies.
- Export settings and numerical voltage samples.

The [pulse and local-SWC workbench](https://nalin-dhiman.github.io/passive-dendritic-tree/legacy.html)
remains available separately. Its backward-Euler transients and arbitrary uploads
have a different numerical scope from the harmonic explorer.

## Run locally

No build, installation or external JavaScript library is required:

```sh
python3 -m http.server 8080 --directory docs
```

Visit `http://localhost:8080`. JavaScript modules and workers require HTTP(S);
opening the HTML as a `file:` URL is unsupported. GitHub Pages serves `docs/`
from `main`.

## Anatomical example

The cortical morphology is **Allen Cell Types Database specimen 588402092**, a
mouse visual-cortical layer-5 spiny neuron. The example retains 2,534 soma and
dendritic source points and 2,533 source edges. Axonal points are excluded.

Seven fixed source locations are used for inputs and recordings: source IDs
`1, 1621, 1783, 2165, 1216, 182, 708`. Refinement inserts cable nodes without
moving those physical locations. Selective thinning affects the basal subtree
rooted at source ID `724`; it does not reselect a branch from voltage outcomes.

Uniform thinning multiplies dendritic radii by `0.97`. The selective factor is
`0.8648954759986717` on that subtree. Both remove 5.91% of modeled dendritic
volume while holding the soma fixed. Soma-plus-dendrite volume therefore changes
by a smaller fraction. Material displays distinguish these denominators.

Source: [Allen morphology](https://celltypes.brain-map.org/experiment/morphology/588402092),
[exact SWC](https://api.brain-map.org/api/v2/well_known_file_download/675965556).
Allen-derived data are supplied under its terms for attributed noncommercial
use; see [NOTICES.md](NOTICES.md).

## Electrical model

Coordinates and radii are in micrometres. Internal edges are linear-radius
frusta. A single equipotential spherical soma contributes fixed membrane area;
its stem attachments use the child's dendritic radius. Terminal cable ends are
sealed; spines, junction ledges and end-cap membrane are omitted.

For length `L` and endpoint radii `r0, r1`:

```text
volume = pi L (r0^2 + r0 r1 + r1^2) / 3            [um^3]
lateral area = pi (r0 + r1) sqrt(L^2 + (r1-r0)^2)  [um^2]
axial resistance = 0.01 Ra L / (pi r0 r1)          [MOhm]
```

Exact half-frustum membrane areas are assigned to adjacent node control volumes.
With nodal leak matrix `G`, capacitance `C` and axial coupling included in `G`,
the harmonic equation is

```text
(G + i 2 pi f C / 1000) Vhat = Ihat
V(t) = E_L + Re[Vhat exp(i 2 pi f t / 1000)]
```

Here time is in ms, frequency in Hz, voltage in mV, current in nA, conductance
in uS and capacitance in nF. The UI accepts pA and converts to nA. Nominal
parameters are `Ra=100 Ohm cm`, `gm=0.0001 S/cm^2`, `cm=1 uF/cm^2` and
`E_L=-65 mV`. A complex tree-elimination solver computes both original and
modified responses with identical input and physical recording locations.

## Reading the comparisons

Let `Z0` and `Zc` be the original and modified seven-port impedance matrices,
and `D=Zc-Z0`. The main explorer uses these three distinct measures:

```text
somatic gain discrepancy = ||D[soma,:]||_2 / ||Z0||_2
distributed gain discrepancy = ||D||_2 / ||Z0||_2
complete-response-relative discrepancy = ||D inverse(Z0)||_2
```

The matrix norm is the spectral norm. The last measure normalizes each response
difference against the original response to the same current. These matrix
comparisons allow signed and phase-controlled currents across the seven ports;
they differ from the displayed single-input voltage trace. The 5% guide is an
engineering comparison margin. Sampled frequencies and seven ports do not cover
every possible input, location or continuous frequency.

## Numerical scope

The harmonic implementation is checked against saved independent sparse-matrix
solutions, analytical fixtures, current balance and refinement. Test descriptions
identify the conditions covered. Intermediate frequencies are live calculations,
not an assertion that every frequency has independently certified accuracy.

An earlier reconstructed distal rectangular-current comparison exceeded its
declared 0.0025 mV temporal accuracy requirement. That failure remains unchanged;
harmonic verification does not certify that transient family. The separate legacy
pulse workbench is an educational finite-step calculation.

All examples are passive electrical models. They contain no active channels,
conductance-based synapses, ATP accounting or morphology optimization. Displayed
material describes modeled geometry, with reconstruction and radius uncertainty.

## Tests and reusable Python code

```sh
npm test
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

`python/passive_tree.py` supplies frustum geometry, axial resistance, sparse
frequency-domain impedance, an independent tree solve and Crank–Nicolson
transients. Its reader retains soma and dendritic types 1, 3 and 4. Python
transient voltages are relative to rest; the legacy browser plots absolute
voltage. These remain separate numerical implementations with explicit policies.
