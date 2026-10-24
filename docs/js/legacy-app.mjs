import {makeY, parseSwc, refineGeometry} from './engine.mjs';

const $ = id => document.getElementById(id);
const colors = {root: '#c66f4e', a: '#147f6d', b: '#8463b9', ink: '#143039', muted: '#71858a'};
const state = {geometry: makeY(), results: null, view: 'modified', time: 0, playing: false, request: 0, rendered: 0, lastClock: null, filename: ''};
const controls = ['amplitude', 'duration', 'uniform', 'branch', 'input-site'];
const worker = new Worker(new URL('./simulation-worker.mjs', import.meta.url), {type: 'module'});
let debounce, pendingAutoplay = true;

function context(canvas) {
  const {width, height} = canvas.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}
function number(value) { return new Intl.NumberFormat('en', {maximumFractionDigits: 0}).format(value); }
function setPlaying(value) {
  state.playing = value; state.lastClock = null;
  $('play').innerHTML = `<span aria-hidden="true">${value ? 'Ⅱ' : '▶'}</span> ${value ? 'Pause' : 'Play'}`;
  $('play').setAttribute('aria-label', value ? 'Pause voltage animation' : 'Play voltage animation');
}
function updateOutputs() {
  $('amplitude-value').textContent = `${Number($('amplitude').value).toFixed(2)} nA`;
  $('duration-value').textContent = `${$('duration').value} ms`;
  $('uniform-value').textContent = `${Math.round(100 * Number($('uniform').value))}%`;
  $('branch-value').textContent = `${Math.round(100 * Number($('branch').value))}%`;
}
function requestSimulation() {
  clearTimeout(debounce); updateOutputs();
  $('simulation-status').textContent = 'Solving baseline and modified cable locally…';
  $('download').disabled = true;
  debounce = setTimeout(() => {
    setPlaying(false); state.time = 0;
    worker.postMessage({id: ++state.request, geometry: state.geometry,
      scales: {uniform: Number($('uniform').value), branch: Number($('branch').value)},
      config: {site: $('input-site').value, amplitude: Number($('amplitude').value), duration: Number($('duration').value)}});
  }, 100);
}
worker.onmessage = ({data}) => {
  if (data.id !== state.request) return;
  if (data.error) { $('simulation-status').textContent = data.error; $('simulation-status').classList.add('error'); return; }
  state.results = data; state.rendered = data.id; state.time = 0;
  $('simulation-status').classList.remove('error');
  $('simulation-status').textContent = 'Computed in your browser · 0.05 ms timestep · 0.2 ms displayed samples';
  $('mesh-count').textContent = `${number(data.modified.n)} compartments`;
  $('volume').textContent = `${number(data.modified.cableVolume)} µm³`;
  $('area').textContent = `${number(data.modified.totalArea - data.modified.somaArea)} µm²`;
  const dv = 100 * (data.modified.cableVolume / data.baseline.cableVolume - 1), da = 100 * ((data.modified.totalArea - data.modified.somaArea) / (data.baseline.totalArea - data.baseline.somaArea) - 1);
  $('material-change').textContent = Math.abs(dv) < 0.0001 ? 'Baseline and modified cable material are identical.' : `${dv > 0 ? '+' : '−'}${Math.abs(dv).toFixed(2)}% cable volume · ${da > 0 ? '+' : '−'}${Math.abs(da).toFixed(2)}% cable area, relative to baseline.`;
  $('download').disabled = false; $('scrubber').value = 0;
  draw();
  if (pendingAutoplay && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) { pendingAutoplay = false; setPlaying(true); }
};
worker.onerror = () => {
  $('simulation-status').textContent = 'The simulation worker could not start. Serve these files over HTTP(S), rather than opening index.html directly.';
  $('simulation-status').classList.add('error');
};

function voltageColor(v, min, max) {
  const stops = [[56, 93, 138], [67, 165, 150], [224, 202, 125], [231, 129, 85]];
  const t = Math.max(0, Math.min(1, (v - min) / Math.max(1, max - min))) * 3, i = Math.min(2, Math.floor(t)), u = t - i;
  return `rgb(${stops[i].map((c, k) => Math.round(c * (1 - u) + stops[i + 1][k] * u)).join(',')})`;
}
function currentFrame(trace) { return Math.min(trace.count - 1, Math.round(state.time / trace.config.sampleEvery)); }
function voltageRange() {
  if (!state.results) return {min: -65, max: -45};
  const extrema = [];
  for (const key of ['baseTrace', 'modifiedTrace']) for (const port of ['root', 'a', 'b']) {
    const values = state.results[key][port];
    extrema.push(values.reduce((a, b) => Math.min(a, b), -65), values.reduce((a, b) => Math.max(a, b), -65));
  }
  const low = Math.min(...extrema), high = Math.max(...extrema);
  return {min: Math.floor(Math.min(-65, low) / 5) * 5, max: Math.ceil(Math.max(-60, high) / 5) * 5};
}
function project(width, height) {
  const nodes = state.geometry.nodes, dimensions = ['x', 'y', 'z'].map(key => ({key, min: Math.min(...nodes.map(n => n[key])), max: Math.max(...nodes.map(n => n[key]))}));
  const order = state.geometry.source === 'synthetic' ? [dimensions[0], dimensions[1]] : [...dimensions].sort((a, b) => (b.max - b.min) - (a.max - a.min)).slice(0, 2);
  const [x, y] = order, padX = width < 400 ? 56 : 90, padY = 44;
  const scale = Math.min((width - padX * 2) / Math.max(1, x.max - x.min), (height - padY * 2) / Math.max(1, y.max - y.min));
  return {points: nodes.map(n => ({x: width / 2 + (n[x.key] - (x.min + x.max) / 2) * scale, y: height / 2 - (n[y.key] - (y.min + y.max) / 2) * scale})), axes: `${x.key}${y.key}`, scale};
}
function drawTree() {
  const {ctx, width, height} = context($('tree'));
  if (!state.results || width < 10) return;
  const trace = state.view === 'modified' ? state.results.modifiedTrace : state.results.baseTrace, model = state.results[state.view];
  const frame = currentFrame(trace), {points, axes, scale} = project(width, height), range = voltageRange();
  for (const edge of model.geometry) {
    const a = points[edge.parent], b = points[edge.child], v = (trace.frames[frame * trace.n + edge.parent] + trace.frames[frame * trace.n + edge.child]) / 2;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.lineCap = 'round';
    ctx.strokeStyle = voltageColor(v, range.min, range.max); ctx.lineWidth = Math.max(1.1, Math.min(13, (edge.r0 + edge.r1) * scale * 1.8)); ctx.stroke();
  }
  const named = [['root', 'Root'], ['a', 'A'], ['b', 'B']];
  for (const [key, label] of named) {
    const p = points[state.geometry.ports[key]], radius = key === 'root' ? 5.5 : 4;
    ctx.beginPath(); ctx.arc(p.x, p.y, radius + 3, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
    ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fillStyle = colors[key]; ctx.fill();
    let labelX = p.x + (p.x < width / 2 ? -12 : 12), align = p.x < width / 2 ? 'right' : 'left';
    if (key === 'root' && state.geometry.source === 'synthetic') {labelX = p.x + 13; align = 'left';}
    ctx.font = '600 11px system-ui, sans-serif'; ctx.fillStyle = colors[key]; ctx.textAlign = align; ctx.fillText(label, labelX, p.y + 4);
  }
  const port = $('input-site').value, q = points[state.geometry.ports[port]], active = state.time >= 20 && state.time < 20 + Number($('duration').value);
  if (active) {ctx.beginPath(); ctx.arc(q.x, q.y, 15, 0, Math.PI * 2); ctx.strokeStyle = '#cfad53'; ctx.lineWidth = 1.5; ctx.stroke();}
  const arrowDirection = q.x > width / 2 + 10 ? -1 : 1;
  ctx.strokeStyle = active ? '#b18a27' : '#768b8c'; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(q.x + arrowDirection * 35, q.y + 18); ctx.lineTo(q.x + arrowDirection * 10, q.y + 4);
  ctx.lineTo(q.x + arrowDirection * 15, q.y + 12); ctx.moveTo(q.x + arrowDirection * 10, q.y + 4); ctx.lineTo(q.x + arrowDirection * 20, q.y + 4); ctx.stroke();
  ctx.textAlign = 'left'; ctx.font = '9px system-ui, sans-serif'; ctx.fillStyle = '#81938f';
  ctx.fillText(`${axes.toUpperCase()} projection · widths exaggerated`, 10, height - 9);
  // A physical centerline scale; radius display is explicitly exaggerated.
  const barLength = 100 * scale;
  if (barLength > 12 && barLength < width / 3) {
    const x = width - barLength - 12, y = height - 20;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + barLength, y); ctx.strokeStyle = '#7e938d'; ctx.lineWidth = 1; ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillText('100 µm', x + barLength / 2, y - 5);
  }
  $('scale-min').textContent = `${range.min} mV`; $('scale-max').textContent = `${range.max} mV`;
}
function drawTraces() {
  const {ctx, width, height} = context($('traces'));
  if (!state.results || width < 10) return;
  const left = 48, right = 14, top = 20, bottom = 35, w = width - left - right, h = height - top - bottom;
  const traces = [state.results.baseTrace, state.results.modifiedTrace], all = traces.flatMap(t => [t.root, t.b]);
  const low = Math.min(-65, ...all.map(v => v.reduce((a, b) => Math.min(a, b), -65))), high = Math.max(-65, ...all.map(v => v.reduce((a, b) => Math.max(a, b), -65)));
  const margin = Math.max(0.5, (high - low) * 0.12), min = low - margin, max = high + margin;
  const x = t => left + t / 160 * w, y = v => top + (max - v) / (max - min) * h;
  ctx.fillStyle = '#f6ecd75c'; ctx.fillRect(x(20), top, x(20 + Number($('duration').value)) - x(20), h);
  ctx.font = '9px system-ui, sans-serif'; ctx.fillStyle = colors.muted; ctx.textAlign = 'right';
  for (let k = 0; k <= 4; k++) {
    const v = min + (max - min) * k / 4, py = y(v);
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(width - right, py); ctx.strokeStyle = '#eaf0ec'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillText(v.toFixed(1), left - 9, py + 3);
  }
  ctx.textAlign = 'center';
  for (let t = 0; t <= 160; t += 40) ctx.fillText(String(t), x(t), height - 17);
  ctx.save(); ctx.translate(12, top + h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('Voltage (mV)', 0, 0); ctx.restore();
  ctx.fillText('Time (ms)', left + w / 2, height - 2);
  for (const [i, trace] of traces.entries()) for (const port of ['root', 'b']) {
    ctx.beginPath(); trace.time.forEach((t, j) => {if (j === 0) ctx.moveTo(x(t), y(trace[port][j])); else ctx.lineTo(x(t), y(trace[port][j]));});
    ctx.strokeStyle = colors[port]; ctx.lineWidth = i === 0 ? 1.4 : 2; ctx.globalAlpha = i === 0 ? 0.55 : 1; ctx.setLineDash(i === 0 ? [5, 4] : []); ctx.stroke();
  }
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.moveTo(x(state.time), top); ctx.lineTo(x(state.time), top + h); ctx.strokeStyle = '#9bb6ab'; ctx.lineWidth = 1; ctx.stroke();
}
function draw() {
  $('time-label').textContent = state.time.toFixed(1); $('scrubber').value = state.time;
  drawTree(); drawTraces();
}
function animate(clock) {
  if (state.playing && state.results) {
    if (state.lastClock !== null) state.time = Math.min(160, state.time + Math.min(100, clock - state.lastClock) / 1000 * Number($('speed').value));
    state.lastClock = clock; draw();
    if (state.time >= 160) setPlaying(false);
  }
  requestAnimationFrame(animate);
}
for (const id of controls) $(id).addEventListener(id === 'input-site' ? 'change' : 'input', requestSimulation);
$('play').addEventListener('click', () => {if (state.time >= 160) state.time = 0; setPlaying(!state.playing); draw();});
$('restart').addEventListener('click', () => {state.time = 0; draw();});
$('scrubber').addEventListener('input', () => {setPlaying(false); state.time = Number($('scrubber').value); draw();});
for (const view of ['baseline', 'modified']) $('view-' + view).addEventListener('click', () => {
  state.view = view;
  for (const key of ['baseline', 'modified']) {$('view-' + key).classList.toggle('selected', key === view); $('view-' + key).setAttribute('aria-pressed', String(key === view));}
  draw();
});
$('reset').addEventListener('click', () => {
  $('amplitude').value = 0.15; $('duration').value = 20; $('uniform').value = 1; $('branch').value = 1; $('input-site').value = 'b'; requestSimulation();
});
$('swc-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    if (file.size > 2_000_000) throw new Error('Choose a file smaller than 2 MB.');
    const parsed = parseSwc(await file.text());
    const geometry = refineGeometry(parsed, 15);
    state.geometry = geometry; state.filename = file.name;
    $('model-name').textContent = 'Local SWC';
    $('file-status').classList.remove('error');
    $('file-status').textContent = `${file.name}: ${number(parsed.originalPoints)} original points; ${number(geometry.nodes.length)} cable nodes after refinement. All cable types included. Soma area is fixed and excluded from the cable-material comparison. Only a passive geometric illustration; anatomy and radius calibration are not validated by this upload.`;
    $('restore-y').hidden = false;
    $('branch-help').textContent = 'The selected branch is the terminal branch leading to the farthest cable tip. Both multipliers apply there. Lengths, soma area and membrane densities stay fixed.';
    requestSimulation();
  } catch (error) {$('file-status').textContent = error.message; $('file-status').classList.add('error');}
  event.target.value = '';
});
$('restore-y').addEventListener('click', () => {
  state.geometry = makeY(); state.filename = ''; $('model-name').textContent = 'Ideal Y cable'; $('restore-y').hidden = true;
  $('file-status').textContent = 'A single root soma marker is a sphere; cable attachments use child radii. All cable types are passive. Multi-point somata are not supported.';
  $('file-status').classList.remove('error'); $('branch-help').textContent = 'The selected branch is daughter B. Both multipliers apply there. Lengths and membrane densities stay fixed.';
  requestSimulation();
});
$('download').addEventListener('click', () => {
  if (!state.results) return;
  const {baseTrace: base, modifiedTrace: modified} = state.results;
  const lines = ['time_ms,baseline_root_mV,modified_root_mV,baseline_distal_A_mV,modified_distal_A_mV,baseline_distal_B_mV,modified_distal_B_mV'];
  for (let j = 0; j < base.count; j++) lines.push([base.time[j], base.root[j], modified.root[j], base.a[j], modified.a[j], base.b[j], modified.b[j]].map(v => v.toPrecision(12)).join(','));
  const blob = new Blob([lines.join('\n') + '\n'], {type: 'text/csv'}), url = URL.createObjectURL(blob), anchor = document.createElement('a');
  anchor.href = url; anchor.download = 'passive-cable-traces.csv'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
new ResizeObserver(draw).observe($('tree')); new ResizeObserver(draw).observe($('traces'));
requestSimulation(); requestAnimationFrame(animate);
