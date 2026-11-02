import { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";
import Overview from "./Overview";

const get = (u: string) => fetch(u).then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)));
const post = (u: string, b: any) =>
  fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then((r) => r.json());
const pct = (x?: number | null) => (x == null ? "-" : x.toFixed(3));
const color = (n: any) => (n.label === "illicit" ? "#d62828" : n.label === "licit" ? "#2a9d8f" : "#8d99ae");

function Warn() {
  return <div className="warn">A model score is a prioritisation signal for human review, not proof of criminal activity.</div>;
}

function Dashboard({ models, open }: { models: string[]; open: (id: number) => void }) {
  const [f, setF] = useState({ model: models[0], split: "test", label: "all", predicted: "", min_score: "0" });
  const [res, setRes] = useState<any>(null);
  useEffect(() => {
    const q = new URLSearchParams({ model: f.model, split: f.split, label: f.label, min_score: f.min_score, limit: "50" });
    if (f.predicted) q.set("predicted", f.predicted);
    get("/api/transactions?" + q).then(setRes).catch(() => setRes(null));
  }, [f]);
  const sel = (k: string, opts: string[][]) => (
    <select value={(f as any)[k]} onChange={(e) => setF({ ...f, [k]: e.target.value })}>
      {opts.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
    </select>
  );
  return (
    <>
      <Warn />
      <div className="bar">
        {sel("model", models.map((m) => [m, m]))}
        {sel("split", [["test", "test"], ["val", "val"], ["train", "train"], ["all", "all"]])}
        {sel("label", [["all", "any label"], ["illicit", "illicit"], ["licit", "licit"], ["unknown", "unknown"]])}
        {sel("predicted", [["", "any prediction"], ["true", "flagged"], ["false", "not flagged"]])}
        <input value={f.min_score} size={5} onChange={(e) => setF({ ...f, min_score: e.target.value })} title="min score" />
      </div>
      {res && <p>{res.total} transactions (threshold {pct(res.threshold)}, chosen on validation)</p>}
      <table>
        <thead><tr><th>Tx ID</th><th>Step</th><th>Score</th><th>Flagged</th><th>Known label</th></tr></thead>
        <tbody>
          {res?.items.map((t: any) => (
            <tr key={t.tx_id} className="click" onClick={() => open(t.tx_id)}>
              <td>{t.tx_id}</td><td>{t.time_step}</td><td>{pct(t.score)}</td>
              <td>{t.predicted_illicit ? "yes" : "no"}</td><td>{t.label}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Case({ txId, setTx, models }: { txId: number | null; setTx: (n: number) => void; models: string[] }) {
  const [input, setInput] = useState(txId ? String(txId) : "");
  const [d, setD] = useState<any>(null);
  const [g, setG] = useState<any>(null);
  const [hops, setHops] = useState(1);
  const [notes, setNotes] = useState<any[]>([]);
  const [note, setNote] = useState({ status: "open", note: "" });
  const [err, setErr] = useState("");
  const cyEl = useRef<HTMLDivElement>(null);

  const load = () => {
    if (!txId) return;
    setErr("");
    get(`/api/transactions/${txId}?model=${models[0]}`).then(setD).catch(() => { setD(null); setErr("Transaction not found"); });
    get(`/api/graph/${txId}?hops=${hops}&model=${models[0]}`).then(setG).catch(() => setG(null));
    get(`/api/notes?tx_id=${txId}`).then(setNotes).catch(() => {});
  };
  useEffect(load, [txId, hops]);
  useEffect(() => {
    if (!g || !cyEl.current) return;
    const cy = cytoscape({
      container: cyEl.current,
      elements: [
        ...g.nodes.map((n: any) => ({ data: { id: String(n.tx_id), label: n.score.toFixed(2), c: color(n), root: n.tx_id === txId ? 1 : 0 } })),
        ...g.edges.map((e: any) => ({ data: { source: String(e.source), target: String(e.target) } })),
      ],
      style: [
        { selector: "node", style: { "background-color": "data(c)", label: "data(label)", "font-size": 9 } },
        { selector: "node[root = 1]", style: { "border-width": 3, "border-color": "#fca311" } },
        { selector: "edge", style: { width: 1, "line-color": "#aaa", "target-arrow-shape": "triangle", "target-arrow-color": "#aaa", "curve-style": "bezier" } },
      ],
      layout: { name: "cose", animate: false },
    });
    cy.on("tap", "node", (e: any) => setTx(Number(e.target.id())));
    return () => cy.destroy();
  }, [g]);

  return (
    <>
      <Warn />
      <div className="bar">
        <input placeholder="transaction ID" value={input} onChange={(e) => setInput(e.target.value)} />
        <button onClick={() => input && setTx(Number(input))}>Search</button>
        <label>hops <select value={hops} onChange={(e) => setHops(Number(e.target.value))}>{[1, 2, 3].map((h) => <option key={h}>{h}</option>)}</select></label>
      </div>
      {err && <p>{err}</p>}
      {d && (
        <div className="card">
          <b>Tx {d.tx_id}</b> — step {d.time_step}, split {d.split}, known label: {d.label}
          <p>Scores: {Object.entries(d.all_scores).map(([m, s]: any) => `${m} ${pct(s)}`).join(" | ")} — flagged by {models[0]}: {d.predicted_illicit ? "yes" : "no"}</p>
          <p>Most unusual features (standardised): {Object.entries(d.top_features_standardized).map(([k, v]) => `${k}=${v}`).join(", ")}</p>
          <p>{d.upstream.length} upstream, {d.downstream.length} downstream neighbours (graph colours: red illicit, green licit, grey unknown; node text = score)</p>
        </div>
      )}
      <div id="cy" ref={cyEl} />
      {g?.truncated && <p>Graph truncated for readability.</p>}
      {txId && (
        <div className="card" style={{ marginTop: 14 }}>
          <b>Investigation notes</b> (analyst notes, not ground-truth labels)
          <div className="bar" style={{ marginTop: 8 }}>
            <select value={note.status} onChange={(e) => setNote({ ...note, status: e.target.value })}>
              {["open", "reviewing", "escalated", "dismissed"].map((s) => <option key={s}>{s}</option>)}
            </select>
            <input size={50} placeholder="note" value={note.note} onChange={(e) => setNote({ ...note, note: e.target.value })} />
            <button onClick={() => post("/api/notes", { tx_id: txId, ...note }).then(() => { setNote({ ...note, note: "" }); load(); })}>Save</button>
          </div>
          {notes.map((n) => <div key={n.id}>[{n.status}] {n.note} <small>{n.created_at}</small></div>)}
        </div>
      )}
    </>
  );
}

function Report() {
  const [r, setR] = useState<any>(null);
  useEffect(() => { get("/api/metrics").then(setR); }, []);
  if (!r) return <p>Loading…</p>;
  const names = Object.keys(r.models);
  const best = r.models[names[names.length - 1]];
  return (
    <>
      <p>Temporal split — train {r.split.train}, val {r.split.val}, test {r.split.test}. Illicit-class metrics on labelled test nodes, mean ± std over seeds. Unknown nodes are never counted.</p>
      <table>
        <thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th><th>FP / FN (best run)</th><th>Train s</th></tr></thead>
        <tbody>
          {names.map((m) => {
            const a = r.models[m].test_mean_std, b = r.models[m].best_run;
            const c = (k: string) => `${pct(a[k].mean)} ± ${pct(a[k].std)}`;
            return <tr key={m}><td>{m}</td><td>{c("precision")}</td><td>{c("recall")}</td><td>{c("f1")}</td><td>{c("pr_auc")}</td><td>{b.test.fp} / {b.test.fn}</td><td>{b.train_seconds}</td></tr>;
          })}
        </tbody>
      </table>
      <h4>Per-time-step test F1 (best run per model)</h4>
      <table>
        <thead><tr><th>Step</th>{names.map((m) => <th key={m}>{m}</th>)}</tr></thead>
        <tbody>
          {best.best_run.per_timestep.map((s: any, i: number) => (
            <tr key={s.time_step}><td>{s.time_step}</td>{names.map((m) => <td key={m}>{pct(r.models[m].best_run.per_timestep[i]?.f1)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export default function App() {
  const [tab, setTab] = useState<"about" | "dash" | "case" | "report">("about");
  const [tx, setTx] = useState<number | null>(null);
  const [models, setModels] = useState<string[]>([]);
  useEffect(() => { get("/api/health").then((h) => setModels(h.models.slice().sort((a: string) => (a === "m2mgnn" ? -1 : 0)))); }, []);
  const open = (id: number) => { setTx(id); setTab("case"); };
  const T = (k: any, t: string) => <button className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{t}</button>;
  return (
    <>
      <header><b>GraphGuard</b>{T("about", "Overview")}{T("dash", "Risk dashboard")}{T("case", "Case view")}{T("report", "Model report")}</header>
      <main>
        {tab === "about" ? <Overview go={setTab} /> : !models.length ? <p>Backend not reachable or no trained artifacts found.</p> :
          tab === "dash" ? <Dashboard models={models} open={open} /> :
          tab === "case" ? <Case txId={tx} setTx={setTx} models={models} /> : <Report />}
      </main>
    </>
  );
}