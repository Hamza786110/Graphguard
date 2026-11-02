import { useEffect, useState } from "react";

// EDIT THESE before presenting
const TEAM = ["Hamza " , "Rashi", "Eeshan" ];
const COURSE = "Artificial Intelligence Lab";

const f = (x?: number) => (x == null ? "-" : x.toFixed(3));

const STEPS = [
  ["1. Data", "Elliptic Bitcoin graph: transactions are nodes, money flows are edges. Most nodes are unlabelled."],
  ["2. Leak-free split", "Train on early time steps, validate on the next, test on the latest. No edge crosses time steps."],
  ["3. Models", "XGBoost (features only) vs GCN, GraphSAGE and our M2M-GNN (features + graph)."],
  ["4. Evaluation", "Precision, recall, F1 and PR-AUC on the rare illicit class, over several seeds."],
  ["5. Serving", "Saved weights are loaded by a FastAPI backend. Nothing retrains per request."],
  ["6. Dashboard", "Search a transaction, see its score, its neighbourhood graph and add review notes."],
];

export default function Overview({ go }: { go: (t: "dash" | "case" | "report") => void }) {
  const [r, setR] = useState<any>(null);
  useEffect(() => { fetch("/api/metrics").then((x) => x.json()).then(setR).catch(() => {}); }, []);
  const names: string[] = r ? Object.keys(r.models) : [];
  const best = names.length ? names.reduce((a, b) => (r.models[a].test_mean_std.f1.mean >= r.models[b].test_mean_std.f1.mean ? a : b)) : "";
  const m2m = r?.models?.m2mgnn?.test_mean_std?.f1.mean;
  const bestF1 = best ? r.models[best].test_mean_std.f1.mean : undefined;

  return (
    <div className="about">
      <section className="hero">
        <h1>GraphGuard</h1>
        <p>Graph-based detection of suspicious Bitcoin transactions, built around the M2M-GNN method
          (ICML 2024, <i>Sign is Not a Remedy: Multiset-to-Multiset Message Passing for Learning on Heterophilic Graphs</i>).</p>
        <p className="muted">{COURSE} · {TEAM.join(", ")}</p>
        <div className="bar">
          <button onClick={() => go("dash")}>Open risk dashboard</button>
          <button onClick={() => go("report")}>See model comparison</button>
        </div>
      </section>

      <section>
        <h2>The problem</h2>
        <p>Money-flow graphs link illicit and legitimate transactions together. Standard graph neural networks (GNNs) update a node
          by averaging its neighbours, which works when neighbours are alike and blurs the signal when they differ
          (<b>heterophily</b>). The popular fix, <b>signed message passing</b>, gives different neighbours negative weights, but the paper proves two limits:
          signs that are right at every hop can be wrong over several hops, and class embeddings still converge as layers are added when there are more than two classes.</p>
      </section>

      <section>
        <h2>The paper's solution: M2M-GNN</h2>
        <div className="flow">
          <div className="card">Neighbours<br /><small>each has a feature vector</small></div><span>→</span>
          <div className="card">Soft chunk assignment<br /><small>attention splits every neighbour across chunks</small></div><span>→</span>
          <div className="card">Per-chunk sums<br /><small>groups are aggregated separately</small></div><span>→</span>
          <div className="card">Concatenate<br /><small>different groups are never mixed into one vector</small></div>
        </div>
        <p>A regulariser stops all attention from collapsing into a single chunk. The chunks are learned, so they are not guaranteed to correspond to "illicit" or "licit".</p>
      </section>

      <section>
        <h2>What we built</h2>
        <div className="grid">{STEPS.map(([t, d]) => <div className="card" key={t}><b>{t}</b><p>{d}</p></div>)}</div>
        {r && <p className="muted">Dataset in this run: {r.meta.num_nodes.toLocaleString()} transactions, {r.meta.num_edges_directed.toLocaleString()} edges,
          {" "}{r.meta.n_illicit.toLocaleString()} illicit / {r.meta.n_licit.toLocaleString()} licit / {r.meta.n_unknown.toLocaleString()} unknown labels.
          Test = time steps {r.split.test}.</p>}
      </section>

      <section>
        <h2>Measured results</h2>
        {!r ? <p>Start the backend to load results.</p> : (
          <>
            <table>
              <thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th></tr></thead>
              <tbody>{names.map((m) => { const a = r.models[m].test_mean_std; return (
                <tr key={m} style={m === best ? { fontWeight: 700 } : undefined}>
                  <td>{m}</td><td>{f(a.precision.mean)}</td><td>{f(a.recall.mean)}</td><td>{f(a.f1.mean)} ± {f(a.f1.std)}</td><td>{f(a.pr_auc.mean)}</td></tr>); })}</tbody>
            </table>
            <p>Mean over seeds on the held-out test steps, illicit class only. Best model by F1: <b>{best}</b>
              {m2m != null && best !== "m2mgnn" && <> — M2M-GNN reached F1 {f(m2m)} against {f(bestF1)}, so in our setup it did not outperform the best baseline. We report this as measured.</>}
              {best === "m2mgnn" && <> — M2M-GNN had the highest mean F1 here; check the seed spread before claiming a real difference.</>}</p>
          </>
        )}
      </section>

      <section>
        <h2>Limitations</h2>
        <ul>
          <li>Our M2M-GNN layer was written from the paper's equations; it still has to be compared line by line with the authors' code.</li>
          <li>The paper evaluates heterophilic benchmarks, not fraud. Applying it to Elliptic is our own experiment, with little hyper-parameter tuning.</li>
          <li>Transactions, not accounts. A score is a priority signal for human review, not proof of wrongdoing.</li>
        </ul>
      </section>
    </div>
  );
}