// Water-sound AI (RAVE, "water_pondbrain" model) running off the main thread.
// Messages are handled one at a time, in order.
//   init                 → load the sound-making half (decoder), report its speed
//   dec   {id,z,T}       → T frames of mixed coordinates (frame-major, 16 per frame) → T×2048 samples at 48 kHz, continuing the live stream
//   render{id,z,F}       → a separate, fresh stream rendered all at once (for the chord keys)
//   enc   {k,x}          → a recording (48 kHz) → its coordinates (frame-major, 16 per frame); loads the listening half (encoder) the first time
importScripts('ort.min.js');
ort.env.wasm.wasmPaths = new URL('./', self.location).href;
ort.env.wasm.numThreads = 1;
let dec = null, enc = null, shapes = null, init = null, live = null;
const mk = () => init.map((d, i) => new ort.Tensor('float32', d.slice(), shapes[i]));
function gauss(n, s) { const a = new Float32Array(n); for (let i = 0; i < n; i += 2) { const u = Math.random() || 1e-9, v = Math.random(), r = Math.sqrt(-2 * Math.log(u)) * s; a[i] = r * Math.cos(6.2832 * v); if (i + 1 < n) a[i + 1] = r * Math.sin(6.2832 * v); } return a; }
async function run(zf, T, st) {
  const z = new Float32Array(16 * T); for (let t = 0; t < T; t++) for (let c = 0; c < 16; c++) z[c * T + t] = zf[t * 16 + c];   // to channel-major
  const feeds = { z: new ort.Tensor('float32', z, [1, 16, T]),
                  extra: new ort.Tensor('float32', gauss(112 * T, 0.1), [1, 112, T]),            // the 112 coordinates nobody sets: the AI fills them a little at random
                  rnd: new ort.Tensor('float32', new Float32Array(2 * T * 16 * 64).map(() => Math.random() * 2 - 1), [1, 2 * T, 16, 64]) };
  st.forEach((t, i) => feeds['s' + i] = t);
  const r = await dec.run(feeds);
  return [new Float32Array(r.y.data), st.map((_, i) => r['o' + i])];
}
async function handle(m) {
  if (m.type === 'init') {
    dec = await ort.InferenceSession.create('water_dec.onnx', { executionProviders: ['wasm'] });
    shapes = await (await fetch('state.json')).json();
    const buf = new Float32Array(await (await fetch('state.bin')).arrayBuffer()); let off = 0;
    init = shapes.map(sh => { const n = sh.reduce((a, b) => a * b, 1); const d = buf.slice(off, off + n); off += n; return d; });
    live = mk();
    await run(new Float32Array(16 * 4), 4, mk());
    const t = performance.now(); await run(new Float32Array(16 * 8), 8, mk());
    postMessage({ type: 'ready', ms: (performance.now() - t) / 8 });
  } else if (m.type === 'dec') {
    const [y, ns] = await run(m.z, m.T, live); live = ns;
    postMessage({ type: 'pcm', id: m.id, pcm: y }, [y.buffer]);
  } else if (m.type === 'render') {
    let st = mk(); const out = new Float32Array(m.F * 2048);
    for (let i = 0; i < m.F; i += 8) { const T = Math.min(8, m.F - i); const [y, ns] = await run(m.z.subarray(i * 16, (i + T) * 16), T, st); st = ns; out.set(y, i * 2048); }
    postMessage({ type: 'rendered', id: m.id, pcm: out }, [out.buffer]);
  } else if (m.type === 'enc') {
    if (!enc) enc = await ort.InferenceSession.create('water_enc.onnx', { executionProviders: ['wasm'] });
    const r = await enc.run({ x: new ort.Tensor('float32', m.x, [1, 1, m.x.length]) });
    const F = r.z.dims[2], d = r.z.data, z = new Float32Array(16 * F);
    for (let t = 0; t < F; t++) for (let c = 0; c < 16; c++) z[t * 16 + c] = d[c * F + t];
    postMessage({ type: 'enc', k: m.k, F, z }, [z.buffer]);
  }
}
let q = Promise.resolve();
onmessage = e => { q = q.then(() => handle(e.data)).catch(err => postMessage({ type: 'error', msg: String(err && err.message || err) })); };
