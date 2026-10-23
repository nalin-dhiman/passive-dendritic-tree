import test from 'node:test';
import assert from 'node:assert/strict';
import {frustum, makeY, parseSwc, refineGeometry, buildModel, solveTree, makeStepper, simulate, currentBalance, pulseAt} from '../docs/js/engine.mjs';

const near = (actual, expected, tolerance) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} versus ${expected}; tolerance ${tolerance}`);
function cylinder(step, length = 300, radius = 2) {
  const n = Math.ceil(length / step), nodes = Array.from({length: n + 1}, (_, i) => ({x: length * i / n, y: 0, z: 0}));
  const edges = nodes.slice(1).map((_, i) => ({parent: i, child: i + 1, r0: radius, r1: radius, group: 'a'}));
  return {nodes, edges, somaArea: 0, ports: {root: 0, a: n, b: n}};
}
function denseSolve(matrix, rhs) {
  const a = matrix.map((row, i) => [...row, rhs[i]]), n = a.length;
  for (let k = 0; k < n; k++) {
    const pivot = a[k][k];
    for (let j = k; j <= n; j++) a[k][j] /= pivot;
    for (let i = 0; i < n; i++) if (i !== k) {
      const factor = a[i][k];
      for (let j = k; j <= n; j++) a[i][j] -= factor * a[k][j];
    }
  }
  return a.map(row => row[n]);
}

test('cylinder and frustum geometry retain dimensions and exact analytic integrals', () => {
  const c = frustum(100, 2, 2); near(c.volume, 400 * Math.PI, 1e-10); near(c.area, 400 * Math.PI, 1e-10);
  const f = frustum(3, 1, 5); near(f.volume, 31 * Math.PI, 1e-10); near(f.area, 30 * Math.PI, 1e-10);
  assert.throws(() => frustum(1, 0, 1));
});
test('radius scaling changes volume quadratically and area linearly for cylinders', () => {
  const raw = makeY(), base = buildModel(raw), thin = buildModel(raw, {uniform: 0.8});
  near(thin.cableVolume / base.cableVolume, 0.64, 2e-14);
  near(thin.totalArea / base.totalArea, 0.8, 2e-14);
  near(thin.capacitance.reduce((a, b) => a + b, 0) / base.capacitance.reduce((a, b) => a + b, 0), 0.8, 2e-14);
});
test('linear-radius exact axial resistance and area are invariant to subdivision', () => {
  const raw = {nodes: [{x: 0, y: 0, z: 0}, {x: 100, y: 0, z: 0}], edges: [{parent: 0, child: 1, r0: 1, r1: 3, group: 'b'}], ports: {root: 0, a: 1, b: 1}, somaArea: 0};
  const coarse = buildModel(raw), fine = buildModel(refineGeometry(raw, 7));
  near(fine.totalArea, coarse.totalArea, 1e-10); near(fine.cableVolume, coarse.cableVolume, 1e-10);
  near(fine.axial.slice(1).reduce((r, g) => r + 1 / g, 0), 1 / coarse.axial[1], 1e-10);
  near(1 / coarse.axial[1], 0.01 * 150 * 100 / (Math.PI * 1 * 3), 1e-12);
});
test('membrane current-density and capacitance units yield the specified membrane time constant', () => {
  const m = buildModel(cylinder(20));
  m.capacitance.forEach((c, i) => near(c / m.leak[i], 10, 1e-12));
  near(m.totalArea, 2 * Math.PI * 2 * 300, 1e-9);
});
test('one-compartment RC agrees with the discrete and continuous known answers', () => {
  const model = {n: 1, parent: new Int32Array([-1]), axial: new Float64Array([0]), diag: new Float64Array([0.01]), leak: new Float64Array([0.01]), capacitance: new Float64Array([0.1])};
  const dt = 0.01, stepper = makeStepper(model, dt), count = 1000, current = 0.1;
  let u; for (let i = 0; i < count; i++) u = stepper.step(0, current);
  const discrete = current / 0.01 * (1 - (1 / (1 + dt / 10)) ** count), analytic = 10 * (1 - Math.exp(-1));
  near(u[0], discrete, 5e-12); near(u[0], analytic, 0.002);
});
test('backward-Euler temporal refinement contracts error on independent RC solution', () => {
  const model = {n: 1, parent: new Int32Array([-1]), axial: new Float64Array([0]), diag: new Float64Array([0.01]), leak: new Float64Array([0.01]), capacitance: new Float64Array([0.1])};
  const answer = 10 * (1 - Math.exp(-1)), errors = [0.1, 0.05, 0.025].map(dt => {
    const s = makeStepper(model, dt); let u; for (let i = 0; i < Math.round(10 / dt); i++) u = s.step(0, 0.1); return Math.abs(u[0] - answer);
  });
  assert.ok(errors[0] / errors[1] > 1.98 && errors[1] / errors[2] > 1.98);
});
test('tree elimination agrees with independently assembled dense linear algebra', () => {
  const m = buildModel(makeY(200)), dt = 0.05, diagonal = m.diag.map((v, i) => v + m.capacitance[i] / dt), rhs = Array.from({length: m.n}, (_, i) => Math.sin(i) * 0.1);
  const matrix = Array.from({length: m.n}, (_, i) => Array.from({length: m.n}, (_, j) => i === j ? diagonal[i] : 0));
  for (let i = 1; i < m.n; i++) {matrix[i][m.parent[i]] = -m.axial[i]; matrix[m.parent[i]][i] = -m.axial[i];}
  const dense = denseSolve(matrix, rhs), tree = solveTree(m, diagonal, rhs);
  tree.forEach((v, i) => near(v, dense[i], 2e-12));
});
test('each node and whole cell satisfy capacitive + leak + axial current balance', () => {
  const m = buildModel(makeY(20)), dt = 0.05, s = makeStepper(m, dt), input = m.raw.ports.b;
  for (let k = 0; k < 80; k++) {
    const old = Float64Array.from(s.state()), injected = new Float64Array(m.n); injected[input] = 0.15;
    const u = s.step(input, 0.15), r = currentBalance(m, u, old, dt, injected);
    assert.ok(Math.max(...r.map(Math.abs)) < 1e-12);
    const total = u.reduce((sum, v, i) => sum + m.capacitance[i] * (v - old[i]) / dt + m.leak[i] * v, 0);
    near(total, 0.15, 2e-12);
  }
});
test('cylinder steady state converges quadratically toward sealed-end continuum solution', () => {
  const length = 300, radius = 2, Ra = 150, gm = 1e-4, rcm = radius * 1e-4, lcm = length * 1e-4;
  const K = Math.PI * rcm ** 2 / Ra, q = Math.sqrt(2 * Math.PI * rcm * gm / K);
  const continuumRoot = 0.1 * 1e-6 / (K * q * Math.tanh(q * lcm));
  const continuumTip = 0.1 * 1e-6 / (K * q * Math.sinh(q * lcm));
  const errors = [30, 15, 7.5].map(step => {
    const m = buildModel(cylinder(step, length, radius)), rhs = new Float64Array(m.n); rhs[0] = 0.1;
    const u = solveTree(m, m.diag, rhs);
    if (step === 7.5) {near(u[0], continuumRoot, 0.001); near(u[m.n - 1], continuumTip, 0.001);}
    return Math.abs(u[0] - continuumRoot);
  });
  assert.ok(errors[0] / errors[1] > 3.9 && errors[1] / errors[2] > 3.9);
});
test('Y daughters are electrically symmetric and geometry is not relabeled by thinning', () => {
  const m = buildModel(makeY(20)), rhsA = new Float64Array(m.n), rhsB = new Float64Array(m.n);
  rhsA[m.raw.ports.a] = 0.1; rhsB[m.raw.ports.b] = 0.1;
  const a = solveTree(m, m.diag, rhsA), b = solveTree(m, m.diag, rhsB);
  near(a[0], b[0], 1e-11); near(a[m.raw.ports.a], b[m.raw.ports.b], 1e-11);
  const thin = buildModel(m.raw, {branch: 0.7});
  assert.deepEqual(thin.raw.ports, m.raw.ports); assert.equal(thin.n, m.n);
});
test('pulse sampling, rest preservation and terminal recording are explicit', () => {
  near(pulseAt(20, {amplitude: 0.15}), 0.15, 0); near(pulseAt(40), 0, 0);
  const m = buildModel(makeY(50)), result = simulate(m, {end: 60, amplitude: 0});
  assert.equal(result.time[0], 0); assert.equal(result.time[result.count - 1], 60);
  assert.ok(result.root.every(v => v === -65)); assert.ok(result.b.every(v => v === -65));
  const active = simulate(m, {end: 60}); assert.ok(Math.max(...active.b) > -65); assert.ok(active.b[active.b.length - 1] < Math.max(...active.b));
});
test('SWC keeps identities, handles soma attachment explicitly and preserves material when refined', () => {
  const raw = parseSwc('# fixture\n3 3 20 0 0 1 2\n1 1 0 0 0 5 -1\n2 3 10 0 0 1 1\n4 3 10 10 0 1 2\n');
  assert.equal(raw.nodes[0].id, 1); near(raw.somaArea, 100 * Math.PI, 1e-10);
  assert.equal(raw.edges[0].r0, 1); assert.equal(raw.edges[0].r1, 1);
  const a = buildModel(raw), b = buildModel(refineGeometry(raw, 3));
  near(a.cableVolume, b.cableVolume, 1e-11); near(a.totalArea, b.totalArea, 1e-11);
  near(buildModel(raw, {uniform: 0.8}).somaArea, raw.somaArea, 0);
});
test('invalid SWCs fail clearly instead of silently repairing or truncating anatomy', () => {
  for (const text of [
    '1 1 0 0 0 1 -1\n2 3 1 0 0 0 1',
    '1 1 0 0 0 1 -1\n2 3 0 0 0 1 1',
    '1 1 0 0 0 1 -1\n1 3 1 0 0 1 1',
    '1 1 0 0 0 1 -1\n2 3 1 0 0 1 -1',
    '1 1 0 0 0 1 -1\n2 3 1 0 0 1 3\n3 3 2 0 0 1 2',
    '1 1 0 0 0 1 -1\n2 1 1 0 0 1 1',
    '1 1 0 0 0 1 -1\n2 3 1 0 0 1 999',
  ]) assert.throws(() => parseSwc(text));
  assert.throws(() => parseSwc('1 1 0 0 0 1 -1\n2 3 1 0 0 1 1', 1));
});
