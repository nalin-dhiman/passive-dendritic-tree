/** Passive cable finite volumes. Length: µm; time: ms; voltage: mV; current: nA.
 * Each graph edge is a linear-radius frustum. Node control volumes contain
 * half of each incident edge's lateral membrane. No end-cap membrane is added.
 */
export const DEFAULTS = Object.freeze({ra: 150, cm: 1, gm: 0.0001, rest: -65});

export function frustum(length, r0, r1) {
  if (![length, r0, r1].every(Number.isFinite) || length <= 0 || r0 <= 0 || r1 <= 0) {
    throw new Error('Cable lengths and radii must be finite and positive.');
  }
  return {
    volume: Math.PI * length * (r0 * r0 + r0 * r1 + r1 * r1) / 3,
    area: Math.PI * (r0 + r1) * Math.hypot(length, r1 - r0),
  };
}

export function makeY(step = 10) {
  if (!Number.isFinite(step) || step <= 0) throw new Error('Invalid mesh spacing.');
  const nodes = [{x: 0, y: 0, z: 0}], edges = [];
  function section(start, end, radius, group) {
    const a = nodes[start], length = Math.hypot(end.x - a.x, end.y - a.y), count = Math.ceil(length / step);
    let parent = start;
    for (let k = 1; k <= count; k++) {
      const node = nodes.length, q = k / count;
      nodes.push({x: a.x + (end.x - a.x) * q, y: a.y + (end.y - a.y) * q, z: 0});
      edges.push({parent, child: node, r0: radius, r1: radius, group});
      parent = node;
    }
    return parent;
  }
  const junction = section(0, {x: 0, y: 300}, 2, 'parent');
  const a = section(junction, {x: -240, y: 620}, 1.5, 'a');
  const b = section(junction, {x: 240, y: 620}, 1.5, 'b');
  return {nodes, edges, ports: {root: 0, a, b}, somaArea: 0, name: 'Ideal Y cable', source: 'synthetic'};
}

/** Strict, single-root SWC parser. Coordinates and radius must be in µm.
 * A single root type-1 soma is modeled as a sphere. Its outgoing cables start
 * at the child's radius: the soma radius is never used as a cable radius.
 */
export function parseSwc(text, maxNodes = 2500) {
  if (typeof text !== 'string' || text.length > 2_000_000) throw new Error('Choose an SWC file smaller than 2 MB.');
  const records = [], ids = new Set();
  for (const [lineNo, line] of text.split(/\r?\n/).entries()) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const fields = trimmed.split(/\s+/);
    if (fields.length !== 7) throw new Error(`Line ${lineNo + 1}: expected seven SWC columns.`);
    const [id, type, x, y, z, r, parentId] = fields.map(Number);
    if (![id, type, x, y, z, r, parentId].every(Number.isFinite) || !Number.isSafeInteger(id) || id <= 0 || !Number.isSafeInteger(type) || type < 0 || !Number.isSafeInteger(parentId) || r <= 0) {
      throw new Error(`Line ${lineNo + 1}: invalid identity, coordinates or positive radius.`);
    }
    if (ids.has(id)) throw new Error(`Line ${lineNo + 1}: duplicate node ID ${id}.`);
    ids.add(id); records.push({id, type, x, y, z, r, parentId});
    if (records.length > maxNodes) throw new Error(`This interactive demo accepts at most ${maxNodes} SWC points; the file is not truncated.`);
  }
  if (records.length < 2) throw new Error('An SWC needs at least two connected points.');
  const roots = records.filter(n => n.parentId === -1);
  if (roots.length !== 1) throw new Error('Exactly one root with parent -1 is required.');
  const root = roots[0], byId = new Map(records.map(n => [n.id, n])), children = new Map(records.map(n => [n.id, []]));
  for (const n of records) {
    if (n === root) continue;
    if (!byId.has(n.parentId) || n.parentId === n.id) throw new Error(`Node ${n.id}: invalid parent.`);
    if (n.type === 1) throw new Error('Only a single root soma marker is supported; multi-point somata need an explicitly defined model.');
    children.get(n.parentId).push(n);
  }
  const ordered = [root];
  for (let i = 0; i < ordered.length; i++) ordered.push(...children.get(ordered[i].id));
  if (ordered.length !== records.length) throw new Error('SWC contains a disconnected component or cycle.');
  const index = new Map(ordered.map((n, i) => [n.id, i]));
  const nodes = ordered.map(n => ({...n, x: n.x - root.x, y: n.y - root.y, z: n.z - root.z}));
  const distance = new Float64Array(nodes.length), edgeForChild = new Map();
  const edges = nodes.slice(1).map((n, j) => {
    const child = j + 1, parent = index.get(n.parentId), p = nodes[parent];
    const length = Math.hypot(n.x - p.x, n.y - p.y, n.z - p.z);
    if (length <= 1e-9) throw new Error(`Node ${n.id}: zero-length edges are unsupported; no automatic repair is performed.`);
    distance[child] = distance[parent] + length;
    const edge = {parent, child, r0: p.type === 1 ? n.r : p.r, r1: n.r, group: 'other'};
    edgeForChild.set(child, edge); return edge;
  });
  const terminals = nodes.map((n, i) => ({i, d: distance[i]})).filter(t => !children.get(nodes[t.i].id).length).sort((a, b) => b.d - a.d);
  const b = terminals[0].i;
  let ancestor = b;
  while (ancestor !== 0) {
    const edge = edgeForChild.get(ancestor);
    edge.group = 'b';
    if (children.get(nodes[edge.parent].id).length > 1 || edge.parent === 0) break;
    ancestor = edge.parent;
  }
  const a = terminals.find(t => t.i !== b)?.i ?? 0;
  return {nodes, edges, ports: {root: 0, a, b}, somaArea: root.type === 1 ? 4 * Math.PI * root.r * root.r : 0, name: 'Local SWC', source: 'swc', originalPoints: nodes.length};
}

/** Add cable nodes without moving existing physical endpoints or changing radii. */
export function refineGeometry(raw, spacing = 15, maxNodes = 6000) {
  if (!(spacing > 0)) throw new Error('Mesh spacing must be positive.');
  const nodes = [{...raw.nodes[0]}], edges = [], endpoint = new Map([[0, 0]]);
  for (const edge of raw.edges) {
    const p = raw.nodes[edge.parent], q = raw.nodes[edge.child];
    if (!endpoint.has(edge.parent)) throw new Error('Parents must precede their children.');
    const length = Math.hypot(q.x - p.x, q.y - p.y, q.z - p.z), count = Math.ceil(length / spacing);
    let parent = endpoint.get(edge.parent);
    for (let j = 1; j <= count; j++) {
      const u = j / count, child = nodes.length;
      nodes.push({x: p.x + (q.x - p.x) * u, y: p.y + (q.y - p.y) * u, z: p.z + (q.z - p.z) * u});
      edges.push({...edge, parent, child, r0: edge.r0 + (edge.r1 - edge.r0) * (j - 1) / count, r1: edge.r0 + (edge.r1 - edge.r0) * u});
      if (nodes.length > maxNodes) throw new Error(`Refinement would exceed ${maxNodes} compartments; no branches have been dropped.`);
      parent = child;
    }
    endpoint.set(edge.child, parent);
  }
  return {...raw, nodes, edges, ports: Object.fromEntries(Object.entries(raw.ports).map(([k, i]) => [k, endpoint.get(i)]))};
}

export function buildModel(raw, options = {}) {
  const p = {...DEFAULTS, uniform: 1, branch: 1, ...options};
  if (![p.ra, p.cm, p.gm, p.uniform, p.branch].every(v => Number.isFinite(v) && v > 0) || !Number.isFinite(p.rest)) throw new Error('Invalid passive parameters.');
  const n = raw.nodes.length;
  if (!n || raw.edges.length !== n - 1) throw new Error('A connected tree is required.');
  const parent = new Int32Array(n).fill(-1), axial = new Float64Array(n), area = new Float64Array(n), diag = new Float64Array(n);
  const geometry = [], lengthTotal = raw.edges.reduce((s, edge) => {
    const a = raw.nodes[edge.parent], b = raw.nodes[edge.child];
    if (edge.parent >= edge.child || edge.parent < 0 || edge.child >= n || parent[edge.child] !== -1) throw new Error('Expected one parent per node in topological order.');
    const length = Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
    const scale = p.uniform * (edge.group === 'b' ? p.branch : 1), r0 = edge.r0 * scale, r1 = edge.r1 * scale, rm = (r0 + r1) / 2;
    const full = frustum(length, r0, r1);
    area[edge.parent] += frustum(length / 2, r0, rm).area;
    area[edge.child] += frustum(length / 2, rm, r1).area;
    const g = 100 * Math.PI * r0 * r1 / (p.ra * length);
    parent[edge.child] = edge.parent; axial[edge.child] = g;
    diag[edge.parent] += g; diag[edge.child] += g;
    geometry.push({...edge, length, r0, r1, ...full});
    return s + length;
  }, 0);
  if (n > 1 && Array.from(parent.slice(1)).some(v => v < 0)) throw new Error('Disconnected tree.');
  // Soma is fixed during cable-radius changes; it is reported separately.
  area[0] += raw.somaArea || 0;
  const capacitance = area.map(a => a * p.cm * 1e-5); // nF
  const leak = area.map(a => a * p.gm * 0.01); // µS
  for (let i = 0; i < n; i++) diag[i] += leak[i];
  return {raw, parameters: p, n, parent, axial, area, capacitance, leak, diag, geometry, length: lengthTotal,
    totalArea: area.reduce((a, b) => a + b, 0), cableVolume: geometry.reduce((s, edge) => s + edge.volume, 0), somaArea: raw.somaArea || 0};
}

/** Hines elimination, for a symmetric positive-definite tree matrix. */
export function solveTree(model, diagonal, rhs) {
  const d = Float64Array.from(diagonal), b = Float64Array.from(rhs), {n, parent, axial} = model;
  for (let i = n - 1; i > 0; i--) {
    const j = parent[i], g = axial[i];
    if (!(d[i] > 0)) throw new Error('Non-positive tree matrix pivot.');
    d[j] -= g * g / d[i]; b[j] += g * b[i] / d[i];
  }
  if (!(d[0] > 0)) throw new Error('Non-positive root pivot.');
  const v = new Float64Array(n); v[0] = b[0] / d[0];
  for (let i = 1; i < n; i++) v[i] = (b[i] + axial[i] * v[parent[i]]) / d[i];
  return v;
}

export function makeStepper(model, dt = 0.05) {
  if (!Number.isFinite(dt) || dt <= 0) throw new Error('Timestep must be positive.');
  const diagonal = model.diag.map((d, i) => d + model.capacitance[i] / dt), rhs = new Float64Array(model.n);
  let u = new Float64Array(model.n);
  return {
    step(inputNode, inputCurrent) {
      for (let i = 0; i < model.n; i++) rhs[i] = model.capacitance[i] * u[i] / dt;
      rhs[inputNode] += inputCurrent;
      u = solveTree(model, diagonal, rhs); return u;
    },
    state: () => u,
  };
}

export function pulseAt(t, {onset = 20, duration = 20, amplitude = 0.15} = {}) {
  return t >= onset && t < onset + duration ? amplitude : 0;
}

export function simulate(model, config = {}) {
  const p = {dt: 0.05, end: 160, sampleEvery: 0.2, onset: 20, duration: 20, amplitude: 0.15, input: model.raw.ports.b, ...config};
  if (!Number.isInteger(p.input) || p.input < 0 || p.input >= model.n) throw new Error('Invalid input site.');
  if (!(p.end > 0 && p.end <= 1000 && p.dt > 0 && p.sampleEvery >= p.dt) || ![p.onset, p.duration, p.amplitude].every(Number.isFinite) || p.onset < 0 || p.duration < 0) throw new Error('Invalid simulation settings.');
  const steps = Math.round(p.end / p.dt), stride = Math.max(1, Math.round(p.sampleEvery / p.dt));
  const count = Math.floor(steps / stride) + 1 + (steps % stride ? 1 : 0);
  const time = new Float64Array(count), frames = new Float32Array(count * model.n);
  const root = new Float64Array(count), a = new Float64Array(count), b = new Float64Array(count);
  const ports = model.raw.ports, solver = makeStepper(model, p.dt), rest = model.parameters.rest;
  let frame = 0;
  function save(t, u) {
    time[frame] = t; root[frame] = rest + u[ports.root]; a[frame] = rest + u[ports.a]; b[frame] = rest + u[ports.b];
    for (let i = 0; i < model.n; i++) frames[frame * model.n + i] = rest + u[i];
    frame++;
  }
  save(0, solver.state());
  for (let k = 1; k <= steps; k++) {
    const u = solver.step(p.input, pulseAt((k - 0.5) * p.dt, p));
    if (k % stride === 0 || k === steps) save(k * p.dt, u);
  }
  return {time, root, a, b, frames, count: frame, n: model.n, config: p};
}

export function currentBalance(model, u, old, dt, injected) {
  const residual = new Float64Array(model.n);
  for (let i = 0; i < model.n; i++) residual[i] = model.capacitance[i] * (u[i] - old[i]) / dt + model.leak[i] * u[i] - (injected[i] || 0);
  for (let i = 1; i < model.n; i++) {
    const flux = model.axial[i] * (u[i] - u[model.parent[i]]);
    residual[i] += flux; residual[model.parent[i]] -= flux;
  }
  return residual;
}
