// A sound AI (RAVE model, chosen by ?m=water / ?m=voice) running off the main thread. Messages are handled one at a time.
//   init                 → load the sound-making half (decoder, and the prior if the model has one), report its speed
//   dec   {z,T}          → T frames of mixed coordinates (frame-major, 16 per frame) → T×2048 samples at 48 kHz, continuing the live stream
//   render{z,F}          → a separate, fresh stream rendered all at once (for the chord keys)
//   enc   {k,x}          → a recording (48 kHz) → its coordinates (frame-major); loads the listening half (encoder) the first time
importScripts('ort.min.js');
ort.env.wasm.wasmPaths = new URL('./', self.location).href;
ort.env.wasm.numThreads = 1;
const M = new URLSearchParams(self.location.search).get('m') || 'water';
let dec = null, pri = null, enc = null, meta = null, init = null, live = null;
const C = () => meta.crop, X = () => meta.L - meta.crop;
const fresh = () => ({ d: init.d.map((x, i) => new ort.Tensor('float32', x.slice(), meta.dec[i])),
                       p: init.p.map((x, i) => new ort.Tensor('float32', x.slice(), meta.pri[i])) });
function gauss(n, s) { const a = new Float32Array(n); for (let i = 0; i < n; i += 2) { const u = Math.random() || 1e-9, v = Math.random(), r = Math.sqrt(-2 * Math.log(u)) * s; a[i] = r * Math.cos(6.2832 * v); if (i + 1 < n) a[i + 1] = r * Math.sin(6.2832 * v); } return a; }
async function run(zf, T, st) {
  const L = meta.L, c = C(), x = X(), z = new Float32Array(L * T);        // full coordinates, channel-major
  if (pri) {                                                                // models with a prior: it guesses the unset coordinates, frame by frame
    for (let t = 0; t < T; t++) {
      const f = { z: new ort.Tensor('float32', zf.slice(t * c, (t + 1) * c), [1, c, 1]), eps: new ort.Tensor('float32', gauss(x, 0.1), [1, x, 1]) };
      st.p.forEach((s, i) => f['s' + i] = s);
      const r = await pri.run(f); st.p = st.p.map((_, i) => r['o' + i]);
      const fr = r.f.data; for (let k = 0; k < L; k++) z[k * T + t] = fr[k];
    }
  } else {                                                                  // others: the unset coordinates are small random values
    const g = gauss(x * T, 0.1);
    for (let t = 0; t < T; t++) { for (let k = 0; k < c; k++) z[k * T + t] = zf[t * c + k]; for (let k = 0; k < x; k++) z[(c + k) * T + t] = g[k * T + t]; }
  }
  const f = { z: new ort.Tensor('float32', z, [1, L, T]),
              rnd: new ort.Tensor('float32', new Float32Array(2 * T * 16 * 64).map(() => Math.random() * 2 - 1), [1, 2 * T, 16, 64]) };
  st.d.forEach((s, i) => f['s' + i] = s);
  const r = await dec.run(f); st.d = st.d.map((_, i) => r['o' + i]);
  return new Float32Array(r.y.data);
}
async function handle(m) {
  if (m.type === 'init') {
    meta = await (await fetch(M + '_meta.json')).json();
    dec = await ort.InferenceSession.create(M + '_dec.onnx', { executionProviders: ['wasm'] });
    if (meta.pri.length) pri = await ort.InferenceSession.create(M + '_pri.onnx', { executionProviders: ['wasm'] });
    const buf = new Float32Array(await (await fetch(M + '_state.bin')).arrayBuffer()); let off = 0;
    const take = sh => { const n = sh.reduce((a, b) => a * b, 1); const d = buf.slice(off, off + n); off += n; return d; };
    init = { d: meta.dec.map(take), p: meta.pri.map(take) };
    live = fresh();
    await run(new Float32Array(C() * 4), 4, fresh());
    const t = performance.now(); await run(new Float32Array(C() * 8), 8, fresh());
    postMessage({ type: 'ready', ms: (performance.now() - t) / 8 });
  } else if (m.type === 'dec') {
    const y = await run(m.z, m.T, live);
    postMessage({ type: 'pcm', pcm: y }, [y.buffer]);
  } else if (m.type === 'render') {
    const st = fresh(), out = new Float32Array(m.F * 2048), c = C();
    for (let i = 0; i < m.F; i += 8) { const T = Math.min(8, m.F - i); out.set(await run(m.z.subarray(i * c, (i + T) * c), T, st), i * 2048); }
    postMessage({ type: 'rendered', pcm: out }, [out.buffer]);
  } else if (m.type === 'enc') {
    if (!enc) enc = await ort.InferenceSession.create(M + '_enc.onnx', { executionProviders: ['wasm'] });
    const r = await enc.run({ x: new ort.Tensor('float32', m.x, [1, 1, m.x.length]) });
    const c = r.z.dims[1], F = r.z.dims[2], d = r.z.data, z = new Float32Array(c * F);
    for (let t = 0; t < F; t++) for (let k = 0; k < c; k++) z[t * c + k] = d[k * F + t];
    postMessage({ type: 'enc', k: m.k, F, z }, [z.buffer]);
  }
}
let q = Promise.resolve();
onmessage = e => { q = q.then(() => handle(e.data)).catch(err => postMessage({ type: 'error', msg: String(err && err.message || err) })); };
