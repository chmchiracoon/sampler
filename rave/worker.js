// A sound AI (RAVE model, chosen by ?m=water / ?m=voice) running off the main thread. Messages are handled one at a time.
//   init                 → load the sound-making half (decoder, and the prior if the model has one), report its speed
//   dec   {z,T}          → T frames of mixed coordinates (frame-major, 16 per frame) → T×2048 samples at 48 kHz, continuing the live stream
//   render{z,F}          → a separate, fresh stream rendered all at once (for the chord keys)
//   enc   {k,x}          → a recording (48 kHz) → its coordinates (frame-major); loads the listening half (encoder) the first time
importScripts('ort.wasm.min.js');                                          // CPU-only runtime (11 MB instead of 22 MB)
ort.env.wasm.wasmPaths = new URL('./', self.location).href;
ort.env.wasm.numThreads = 1;
const M = new URLSearchParams(self.location.search).get('m') || 'water';
// Big files are kept in the browser's own storage after the first download, so later visits only re-prepare them.
// Bump CACHE when the model files change.
const CACHE = 'rave-2026-10-27a';
async function bytes(url) {
  let c = null; try { c = await caches.open(CACHE); const hit = await c.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch (e) {}
  const r = await fetch(url); if (!r.ok) throw new Error(url + ' ' + r.status);
  const b = await r.arrayBuffer();
  if (c) { try { await c.put(url, new Response(b.slice(0))); } catch (e) {} }
  return new Uint8Array(b);
}
async function session(url) { return ort.InferenceSession.create(await bytes(url), { executionProviders: ['wasm'] }); }
let wasmReady = null;
function prepWasm() { return wasmReady || (wasmReady = bytes('ort-wasm-simd-threaded.wasm').then(b => { ort.env.wasm.wasmBinary = b.buffer; })); }
// clear older versions
caches.keys().then(ks => ks.filter(k => k.startsWith('rave-') && k !== CACHE).forEach(k => caches.delete(k))).catch(() => {});
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
    await prepWasm();
    meta = await (await fetch(M + '_meta.json')).json();
    dec = await session(M + '_dec.onnx');
    if (meta.pri.length) pri = await session(M + '_pri.onnx');
    const sb = await bytes(M + '_state.bin'), buf = new Float32Array(sb.buffer, sb.byteOffset, sb.byteLength / 4); let off = 0;
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
    if (!enc) { await prepWasm(); enc = await session(M + '_enc.onnx'); }
    const r = await enc.run({ x: new ort.Tensor('float32', m.x, [1, 1, m.x.length]) });
    const c = r.z.dims[1], F = r.z.dims[2], d = r.z.data, z = new Float32Array(c * F);
    for (let t = 0; t < F; t++) for (let k = 0; k < c; k++) z[t * c + k] = d[k * F + t];
    postMessage({ type: 'enc', k: m.k, F, z }, [z.buffer]);
  }
}
let q = Promise.resolve();
onmessage = e => { q = q.then(() => handle(e.data)).catch(err => postMessage({ type: 'error', msg: String(err && err.message || err) })); };
