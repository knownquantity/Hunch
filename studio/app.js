/* Hunch Studio front-end. No frameworks, no build step. */

const S = {
  worlds: [],
  payload: null,
  step: 0,
  maxSteps: 0,
  playing: false,
  timer: null,
  speed: 14,
};

const $ = (sel) => document.querySelector(sel);

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

function field(meta, snap, name) {
  const f = meta.fields.find((x) => x.name === name);
  if (!f) return null;
  const v = snap.slice(f.base, f.base + f.len);
  return f.len === 1 ? v[0] : v;
}

/* Sum one agent's rewards. Summing every key would net to zero in a
 * two-player world, which is exactly wrong for a score display. */
function sumRewards(steps, agent) {
  let t = 0;
  for (const s of steps) t += (s.r[agent] || 0);
  return t;
}

function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = Math.max(80, rect.width);
  const H = Math.max(60, rect.height || rect.width * 0.62);
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return { ctx, W, H };
}

function rr(ctx, x, y, w, h, r) {
  const k = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + k, y);
  ctx.arcTo(x + w, y, x + w, y + h, k);
  ctx.arcTo(x + w, y + h, x, y + h, k);
  ctx.arcTo(x, y + h, x, y, k);
  ctx.arcTo(x, y, x + w, y, k);
  ctx.closePath();
}

function text(ctx, str, x, y, opts = {}) {
  ctx.font = `${opts.weight || 500} ${opts.size || 11}px ${opts.mono ? "ui-monospace, Menlo, monospace" : "-apple-system, system-ui, sans-serif"}`;
  ctx.fillStyle = opts.color || "#e6ebf2";
  ctx.textAlign = opts.align || "left";
  ctx.textBaseline = opts.baseline || "alphabetic";
  ctx.fillText(str, x, y);
}

/* ------------------------------------------------------------------ */
/* renderers                                                           */
/* ------------------------------------------------------------------ */

const R = {};

R.dungeon = (ctx, W, H, steps, i, meta) => {
  const st = steps[Math.min(i, steps.length - 1)];
  const s = st.s;
  const hx = field(meta, s, "x");
  const hy = field(meta, s, "y");
  const tile = field(meta, s, "tile");
  const seen = field(meta, s, "seen");
  const hasKey = field(meta, s, "has_key");

  const pad = 10;
  const cell = Math.min((W - pad * 2) / 7, (H - pad * 2) / 7);
  const ox = (W - cell * 7) / 2;
  const oy = (H - cell * 7) / 2;

  const palette = {
    0: "#0e141c", 1: "#2b3547", 2: "#4a1a26", 3: "#3a3116", 4: "#143326", 5: "#2e2040",
  };

  for (let y = 0; y < 7; y++) {
    for (let x = 0; x < 7; x++) {
      const t = tile[y * 7 + x];
      const px = ox + x * cell, py = oy + y * cell;
      ctx.fillStyle = palette[t] || "#131922";
      rr(ctx, px + 1, py + 1, cell - 2, cell - 2, 3);
      ctx.fill();
      if (t === 1) {
        ctx.strokeStyle = "#39455c";
        ctx.lineWidth = 1;
        rr(ctx, px + 2, py + 2, cell - 4, cell - 4, 3);
        ctx.stroke();
      }
      if (seen && seen[y * 7 + x] && t !== 1) {
        ctx.fillStyle = "rgba(122,167,255,.08)";
        rr(ctx, px + 1, py + 1, cell - 2, cell - 2, 3);
        ctx.fill();
      }
      const cx = px + cell / 2, cy = py + cell / 2;
      if (t === 2) { // trap
        ctx.fillStyle = "#ff7a85";
        ctx.beginPath();
        for (let k = 0; k < 3; k++) {
          const a = (k * 2 * Math.PI) / 3 - Math.PI / 2;
          const px2 = cx + Math.cos(a) * cell * 0.2;
          const py2 = cy + Math.sin(a) * cell * 0.2;
          if (k === 0) ctx.moveTo(px2, py2); else ctx.lineTo(px2, py2);
        }
        ctx.closePath();
        ctx.fill();
      }
      if (t === 3) { // key
        ctx.fillStyle = "#ffd166";
        ctx.beginPath();
        ctx.arc(cx - cell * 0.06, cy, cell * 0.11, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillRect(cx + cell * 0.03, cy - cell * 0.035, cell * 0.2, cell * 0.07);
      }
      if (t === 5) { // bait
        ctx.fillStyle = "rgba(199,146,234,.85)";
        ctx.beginPath();
        ctx.moveTo(cx, cy - cell * 0.16);
        ctx.lineTo(cx + cell * 0.15, cy);
        ctx.lineTo(cx, cy + cell * 0.16);
        ctx.lineTo(cx - cell * 0.15, cy);
        ctx.closePath();
        ctx.fill();
      }
      if (t === 4) { // exit
        const open = hasKey === 1;
        ctx.strokeStyle = open ? "#7dd3a0" : "#3b4a5f";
        ctx.lineWidth = 2;
        rr(ctx, px + cell * 0.24, py + cell * 0.18, cell * 0.52, cell * 0.64, 3);
        ctx.stroke();
        if (open) {
          ctx.fillStyle = "rgba(125,211,160,.22)";
          rr(ctx, px + cell * 0.24, py + cell * 0.18, cell * 0.52, cell * 0.64, 3);
          ctx.fill();
        }
      }
    }
  }

  // hero
  const px = ox + hx * cell + cell / 2;
  const py = oy + hy * cell + cell / 2;
  ctx.save();
  ctx.shadowColor = "rgba(122,167,255,.9)";
  ctx.shadowBlur = 14;
  ctx.fillStyle = "#cfe0ff";
  ctx.beginPath();
  ctx.arc(px, py, cell * 0.22, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  if (hasKey === 1) {
    ctx.fillStyle = "#ffd166";
    ctx.beginPath();
    ctx.arc(px + cell * 0.2, py - cell * 0.2, cell * 0.07, 0, Math.PI * 2);
    ctx.fill();
  }
};

R.market = (ctx, W, H, steps, i, meta) => {
  const n = Math.min(i + 1, steps.length);
  const mid = [], eq = [], inv = [];
  let cash = 0;
  for (let k = 0; k < n; k++) {
    const s = steps[k].s;
    mid.push(field(meta, s, "mid"));
    cash = field(meta, s, "cash");
    inv.push(field(meta, s, "inv"));
    eq.push(cash + inv[k] * mid[k]);
  }
  const padL = 8, padR = 8, padT = 10, padB = 14;

  const drawSeries = (arr, color, top, height, label) => {
    if (arr.length < 2) return;
    let lo = Math.min(...arr), hi = Math.max(...arr);
    if (hi - lo < 1e-9) { hi = lo + 1; lo -= 1; }
    const xs = (k) => padL + (k / Math.max(1, arr.length - 1)) * (W - padL - padR);
    const ys = (v) => top + height - ((v - lo) / (hi - lo)) * height;
    ctx.beginPath();
    arr.forEach((v, k) => { const X = xs(k), Y = ys(v); k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.lineTo(xs(arr.length - 1), top + height);
    ctx.lineTo(xs(0), top + height);
    ctx.closePath();
    ctx.fillStyle = color.replace("rgb", "rgba").replace(")", ",.10)");
    ctx.fill();
    text(ctx, label, padL + 2, top + 9, { size: 9.5, color: "#8b95a5", mono: true });
  };

  drawSeries(mid, "rgb(122,167,255)", padT, H * 0.36, "price");
  drawSeries(eq, "rgb(125,211,160)", padT + H * 0.42, H * 0.42, "equity");

  const cur = inv.length ? inv[inv.length - 1] : 0;
  const bw = W - padL - padR;
  const cx = padL + bw / 2;
  ctx.fillStyle = "#1b2130";
  rr(ctx, padL, H - 10, bw, 5, 3); ctx.fill();
  const frac = Math.max(-1, Math.min(1, cur / 6));
  ctx.fillStyle = frac >= 0 ? "#7aa7ff" : "#ffb86b";
  const w = Math.abs(frac) * (bw / 2);
  rr(ctx, frac >= 0 ? cx : cx - w, H - 10, Math.max(1, w), 5, 3); ctx.fill();
  text(ctx, `inventory ${cur >= 0 ? "+" : ""}${cur.toFixed(0)}`, W - padR, H - 14,
       { size: 9.5, color: "#8b95a5", mono: true, align: "right" });
  text(ctx, `equity ${eq.length ? eq[eq.length - 1].toFixed(1) : "0"}`, padL, H - 14,
       { size: 9.5, color: "#8b95a5", mono: true });
};

R.duel = (ctx, W, H, steps, i, meta, actions) => {
  const st = steps[Math.min(i, steps.length - 1)];
  const s = st.s;
  const ca = field(meta, s, "card_a");
  const cb = field(meta, s, "card_b");
  const board = field(meta, s, "board");
  const phase = field(meta, s, "phase");
  const bet = field(meta, s, "bet");

  const cardW = Math.min(W * 0.19, 64);
  const cardH = cardW * 1.4;
  const cy = H * 0.36;

  const drawCard = (x, y, label, value, color, hidden) => {
    ctx.fillStyle = "#141a24";
    rr(ctx, x, y, cardW, cardH, 8); ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 1.4;
    rr(ctx, x, y, cardW, cardH, 8); ctx.stroke();
    text(ctx, label, x + cardW / 2, y + 15, { size: 9, color: "#8b95a5", align: "center", mono: true });
    text(ctx, hidden ? "?" : String(value), x + cardW / 2, y + cardH * 0.68,
         { size: 22, color, align: "center", weight: 700, mono: true });
  };

  drawCard(W * 0.09, cy, "ALICE", ca, "#7aa7ff", false);
  drawCard(W - W * 0.09 - cardW, cy, "BOB", cb, "#ffb86b", false);
  drawCard(W / 2 - cardW / 2, cy - cardH * 0.12, "BOARD", board, "#7dd3a0", false);

  let msg = "";
  const a = st.a || {};
  if (a.alice !== undefined) msg = `Alice: ${actions[a.alice] || a.alice}`;
  if (a.bob !== undefined) msg = `Bob: ${actions[a.bob] || a.bob}`;
  text(ctx, msg || (phase === 9 ? "showdown" : ""), W / 2, H * 0.84,
       { size: 12, color: "#e6ebf2", align: "center", weight: 600 });

  const pot = 1 + (bet > 0 ? bet : 0);
  text(ctx, `pot ${pot}`, W / 2, H * 0.93, { size: 10.5, color: "#8b95a5", align: "center", mono: true });
};

R.cache = (ctx, W, H, steps, i, meta) => {
  const st = steps[Math.min(i, steps.length - 1)];
  const s = st.s;
  const key = field(meta, s, "cache_key");
  const freq = field(meta, s, "cache_freq");
  const reqArr = field(meta, s, "req");
  const cur = field(meta, s, "cursor");

  const gap = 8;
  const bw = (W - gap * 3) / 4;
  const bh = H * 0.52;
  const top = H * 0.12;

  for (let k = 0; k < 4; k++) {
    const x = k * (bw + gap);
    const hot = Math.min(1, (freq[k] || 0) / 8);
    ctx.fillStyle = "#131922";
    rr(ctx, x, top, bw, bh, 9); ctx.fill();
    ctx.strokeStyle = key[k] === reqArr[Math.min(cur, 59)] ? "#7dd3a0" : "#232a36";
    ctx.lineWidth = key[k] === reqArr[Math.min(cur, 59)] ? 1.8 : 1;
    rr(ctx, x, top, bw, bh, 9); ctx.stroke();

    const label = key[k] < 0 ? "—" : String(key[k]);
    text(ctx, `slot ${k + 1}`, x + bw / 2, top + 16, { size: 9, color: "#8b95a5", align: "center", mono: true });
    text(ctx, label, x + bw / 2, top + bh * 0.52, {
      size: 20, color: key[k] < 0 ? "#3b4a5f" : "#e6ebf2", align: "center", weight: 700, mono: true,
    });
    ctx.fillStyle = "#1b2130";
    rr(ctx, x + 8, top + bh - 14, bw - 16, 5, 3); ctx.fill();
    ctx.fillStyle = "#7dd3a0";
    rr(ctx, x + 8, top + bh - 14, Math.max(1, (bw - 16) * hot), 5, 3); ctx.fill();
  }

  const want = reqArr[Math.min(cur, 59)];
  const hit = (st.r && Object.values(st.r)[0]) > 0;
  text(ctx, `request ${want}${want > 10 ? "  (scan)" : ""}`, 6, H * 0.08,
       { size: 11, color: "#8b95a5", mono: true });
  text(ctx, hit ? "HIT +1" : "miss", W - 6, H * 0.08,
       { size: 11, color: hit ? "#7dd3a0" : "#ff7a85", mono: true, align: "right", weight: 700 });

  // remaining stream
  const bx = 6, by = H * 0.86, bwd = W - 12;
  const prog = Math.min(1, cur / 60);
  ctx.fillStyle = "#1b2130";
  rr(ctx, bx, by, bwd, 5, 3); ctx.fill();
  ctx.fillStyle = "#7aa7ff";
  rr(ctx, bx, by, Math.max(1, bwd * prog), 5, 3); ctx.fill();
  text(ctx, `${cur} / 60 requests`, bx, by - 6, { size: 9.5, color: "#8b95a5", mono: true });
};

/* ------------------------------------------------------------------ */
/* playback                                                            */
/* ------------------------------------------------------------------ */

function players() {
  const p = S.payload;
  return {
    A: p.gen0 ? p.gen0.episodes[0].steps : [],
    B: p.trained ? p.trained.episodes[0].steps : [],
  };
}

function drawAll() {
  const p = S.payload;
  if (!p) return;
  const { A, B } = players();
  const metaA = p.gen0 ? p.gen0 : null;
  const metaB = p.trained ? p.trained : null;
  const render = R[p.id] || (() => {});

  if (metaA && A.length) {
    const { ctx, W, H } = fit($("#cvA"));
    render(ctx, W, H, A, S.step, metaA, p.actions);
    const nameA = (metaA.agents[0] || {}).name || "";
    $("#retA").textContent = `${nameA} ${sumRewards(A.slice(0, S.step + 1), nameA).toFixed(2)}`;
  }
  if (metaB && B.length) {
    const { ctx, W, H } = fit($("#cvB"));
    render(ctx, W, H, B, S.step, metaB, p.actions);
    const nameB = (metaB.agents[0] || {}).name || "";
    $("#retB").textContent = `${nameB} ${sumRewards(B.slice(0, S.step + 1), nameB).toFixed(2)}`;
  }
  $("#tick").textContent = `${S.step + 1} / ${S.maxSteps}`;
}

function setStep(v) {
  S.step = Math.max(0, Math.min(S.maxSteps - 1, v));
  drawAll();
}

function play() {
  if (S.playing) return;
  S.playing = true;
  $("#playBtn").textContent = "Pause";
  const tick = () => {
    if (!S.playing) return;
    if (S.step >= S.maxSteps - 1) { stop(); return; }
    setStep(S.step + 1);
    S.timer = setTimeout(tick, 1000 / S.speed);
  };
  tick();
}

function stop() {
  S.playing = false;
  $("#playBtn").textContent = "Play";
  clearTimeout(S.timer);
}

/* ------------------------------------------------------------------ */
/* rewind moments                                                      */
/* ------------------------------------------------------------------ */

function renderRewind() {
  const p = S.payload;
  const host = $("#rewind");
  host.innerHTML = "";
  const trace = p.trained;
  if (!trace) {
    host.innerHTML = `<div class="hint">No trained policy yet. Hit “Train live”.</div>`;
    return;
  }
  const steps = trace.episodes[0].steps;
  const cards = [];
  steps.forEach((st, idx) => {
    if (!st.q) return;
    for (const agent in st.q) {
      const q = st.q[agent];
      const spread = Math.max(...q) - Math.min(...q);
      cards.push({ idx, agent, q, spread, probs: (st.p || {})[agent] || [] });
    }
  });
  cards.sort((a, b) => b.spread - a.spread);
  const top = cards.slice(0, 4);
  if (!top.length) {
    host.innerHTML = `<div class="hint">No forks recorded.</div>`;
    return;
  }
  const gmin = Math.min(...top.flatMap((c) => c.q));
  const gmax = Math.max(...top.flatMap((c) => c.q));
  const span = Math.max(1e-6, gmax - gmin);

  for (const c of top) {
    const el = document.createElement("div");
    el.className = "forkcard";
    const best = c.q.indexOf(Math.max(...c.q));
    let rows = "";
    c.q.forEach((v, k) => {
      const frac = Math.max(0.02, (v - gmin) / span);
      const col = k === best ? "#7dd3a0" : v < 0 ? "#5b6b80" : "#7aa7ff";
      rows += `<div class="bar ${k === best ? "best" : ""}">
        <div class="lab">${(p.actions[k] || "action " + k)}</div>
        <div class="track"><div class="fill" style="width:${(frac * 100).toFixed(1)}%;background:${col}"></div></div>
        <div class="val">${v.toFixed(2)}</div>
      </div>`;
    });
    el.innerHTML = `<div class="top"><b>step ${c.idx + 1}</b><span>${c.agent}</span>
        <span style="margin-left:auto">forked ${c.q.length} ways</span></div>
      <div class="bars">${rows}</div>`;
    host.appendChild(el);
  }
}

/* ------------------------------------------------------------------ */
/* learning curve                                                      */
/* ------------------------------------------------------------------ */

function renderCurve(liveCurve) {
  const cv = $("#curve");
  const { ctx, W, H } = fit(cv);
  const bench = S.payload && S.payload.bench;

  const colors = { random: "#5b6b80", reinforce: "#c792ea", ppo: "#ffb86b", fork: "#7dd3a0" };
  const series = [];

  if (bench && bench.algos) {
    for (const algo in bench.algos) {
      const runs = bench.algos[algo].runs || [];
      if (!runs.length) continue;
      const first = runs[0].curve || [];
      const pts = first.map((pt, k) => {
        let sum = 0, c = 0;
        for (const r of runs) {
          const q = (r.curve || [])[k];
          if (q) { sum += q.return; c++; }
        }
        return { step: pt.step, ret: c ? sum / c : 0 };
      });
      series.push({ name: algo, pts, color: colors[algo] || "#888" });
    }
  }
  if (liveCurve && liveCurve.length) {
    const pts = liveCurve
      .map((p) => ({ step: Number(p.step), ret: Number(p.return !== undefined ? p.return : p.ret) }))
      .filter((p) => isFinite(p.step) && isFinite(p.ret));
    if (pts.length) series.push({ name: "live", pts, color: "#ffffff" });
  }

  if (!series.length) {
    ctx.fillStyle = "#8b95a5";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("no benchmark data yet", 10, H / 2);
    return;
  }

  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const s of series) for (const p of s.pts) {
    if (!isFinite(p.step) || !isFinite(p.ret)) continue;
    xmin = Math.min(xmin, p.step); xmax = Math.max(xmax, p.step);
    ymin = Math.min(ymin, p.ret); ymax = Math.max(ymax, p.ret);
  }
  if (!isFinite(xmin)) return;
  if (xmax - xmin < 1) xmax = xmin + 1;
  if (ymax - ymin < 1e-6) { ymax += 1; ymin -= 1; }
  const padY = (ymax - ymin) * 0.12;
  ymin -= padY; ymax += padY;

  const L = 34, Rr = 8, T = 10, B = 16;
  const xs = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - Rr);
  const ys = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);

  ctx.strokeStyle = "#1b2130";
  ctx.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const y = T + (k / 3) * (H - T - B);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - Rr, y); ctx.stroke();
  }
  ctx.fillStyle = "#5b6b80";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText(ymax.toFixed(1), 2, T + 8);
  ctx.fillText(ymin.toFixed(1), 2, H - B);

  for (const s of series) {
    if (!s.pts.length) continue;
    ctx.beginPath();
    s.pts.forEach((p, k) => {
      const X = xs(p.step), Y = ys(p.ret);
      k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    });
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.name === "live" ? 2.2 : 1.7;
    ctx.stroke();
  }

  // legend
  let lx = L + 4;
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, H - 9, 8, 3);
    ctx.fillStyle = "#8b95a5";
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillText(s.name, lx + 11, H - 5);
    lx += 16 + ctx.measureText(s.name).width + 10;
  }
}

function renderBench() {
  const host = $("#benchTable");
  const bench = S.payload && S.payload.bench;
  if (!bench || !bench.algos) {
    host.innerHTML = `<div class="hint">Run <code>python3 benchmarks/run_all.py</code> to fill this in.</div>`;
    return;
  }
  const names = ["random", "reinforce", "ppo", "fork"];
  const colors = { random: "#5b6b80", reinforce: "#c792ea", ppo: "#ffb86b", fork: "#7dd3a0" };
  const vals = names.filter((n) => bench.algos[n]).map((n) => bench.algos[n].mean);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = Math.max(1e-9, hi - lo);

  let html = `<div class="brow"><div class="n">algo</div><div class="track" style="background:none"></div><div class="v">mean</div></div>`;
  for (const n of names) {
    const a = bench.algos[n];
    if (!a) continue;
    const frac = 0.06 + 0.94 * ((a.mean - lo) / span);
    html += `<div class="brow">
      <div class="n">${n}</div>
      <div class="track"><div class="fill" style="width:${(frac * 100).toFixed(1)}%;background:${colors[n]}"></div></div>
      <div class="v">${a.mean.toFixed(2)}</div>
    </div>`;
  }
  html += `<div class="hint" style="margin-top:6px">${bench.budget.toLocaleString()} env steps per run · ${bench.seeds} seeds</div>`;
  host.innerHTML = html;
}

/* ------------------------------------------------------------------ */
/* world loading                                                       */
/* ------------------------------------------------------------------ */

async function loadWorlds() {
  const r = await fetch("/api/worlds");
  const data = await r.json();
  S.worlds = data.worlds;
  const nav = $("#worlds");
  nav.innerHTML = "";
  for (const w of S.worlds) {
    const b = document.createElement("button");
    b.dataset.id = w.id;
    b.innerHTML = `<span class="t">${w.id}</span><span class="b">${w.blurb}</span>`;
    b.onclick = () => loadWorld(w.id);
    nav.appendChild(b);
  }
}

async function loadWorld(id) {
  stop();
  document.querySelectorAll("#worlds button").forEach((b) =>
    b.classList.toggle("on", b.dataset.id === id));
  $("#title").textContent = "Loading…";
  const r = await fetch(`/api/world/${id}`);
  const data = await r.json();
  if (data.error) { $("#title").textContent = "Error: " + data.error; return; }
  S.payload = data;
  const { A, B } = players();
  S.maxSteps = Math.max(A.length, B.length, 1);
  S.step = 0;

  $("#title").textContent = data.world;
  $("#blurb").textContent = data.blurb;
  $("#lesson").innerHTML = `<b style="color:#8b95a5">why it matters</b> — ${data.lesson}`;
  $("#seedBadge").textContent = `seed ${data.seed} · one episode`;
  $("#labelA").textContent = "Round 1 · never trained";
  $("#labelB").textContent = data.has_policy ? "After training" : "No policy yet";
  $("#source").innerHTML = highlight(data.source || "");
  $("#legend").textContent = "";

  renderRewind();
  renderBench();
  renderCurve(null);
  drawAll();
  setTimeout(play, 300);
}

function highlight(src) {
  const esc = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc
    .split("\n")
    .map((line) => {
      const c = line.indexOf("//");
      let code = c >= 0 ? line.slice(0, c) : line;
      let com = c >= 0 ? line.slice(c) : "";
      code = code
        .replace(/\b(world|horizon|state|init|step|fn|let|if|else|while|for|in|return|hunch|reward|done|true|false)\b/g,
                 '<span class="kw">$1</span>')
        .replace(/\b(\d+\.?\d*)\b/g, '<span class="num">$1</span>');
      com = com ? `<span class="com">${com}</span>` : "";
      return code + com;
    })
    .join("\n");
}

/* ------------------------------------------------------------------ */
/* controls                                                            */
/* ------------------------------------------------------------------ */

$("#playBtn").onclick = () => (S.playing ? stop() : play());
$("#stepBtn").onclick = () => { stop(); setStep(S.step + 1); };
$("#resetBtn").onclick = () => { stop(); setStep(0); };
$("#speed").oninput = (e) => { S.speed = Number(e.target.value); };

$("#trainBtn").onclick = async () => {
  const btn = $("#trainBtn");
  btn.disabled = true;
  btn.textContent = "Training…";
  $("#trainOut").innerHTML = `<span>running ${$("#algo").value} for ${Number($("#liveBudget").value).toLocaleString()} steps…</span>`;
  try {
    const body = {
      world: S.payload.id,
      algo: $("#algo").value,
      budget: Number($("#liveBudget").value),
      seed: Math.floor(Math.random() * 1000) + 1,
    };
    const t0 = performance.now();
    const r = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("#trainOut").innerHTML =
      `<span class="ok">done in ${secs}s</span> — ${data.steps_used.toLocaleString()} env steps, ` +
      `final return ${data.final_return.toFixed(3)}`;
    if (data.trace) {
      S.payload.trained = data.trace;
      S.payload.has_policy = true;
      const { A, B } = players();
      S.maxSteps = Math.max(A.length, B.length, 1);
      S.step = 0;
    }
    renderRewind();
    renderCurve(data.curve);
    drawAll();
  } catch (e) {
    $("#trainOut").innerHTML = `<span style="color:#ff7a85">${e.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Train live";
  }
};

window.addEventListener("resize", () => { drawAll(); renderCurve(null); });

/* ------------------------------------------------------------------ */
/* boot                                                                */
/* ------------------------------------------------------------------ */

(async function boot() {
  await loadWorlds();
  if (S.worlds.length) await loadWorld(S.worlds[0].id);
  setTimeout(play, 500);
})();
