// src/main.rs
//! BMSSP-style directed SSSP (skeleton) with improved M1 pivots:
//! - FindPivots: k-round local exploration → shortest-path DAG approximation
//!   → pick minimal roots whose forward-subtree size ≥ k as pivots.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap, VecDeque};

use ordered_float::OrderedFloat;

type NodeId = usize;
type Weight = f64;

const EPS: f64 = 1e-12;

/// Represents a directed edge in the graph.
#[derive(Clone, Debug)]
pub struct Edge {
    pub to: NodeId,
    pub w: Weight,
}

/// Directed graph structure with adjacency list representation.
#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub adj: Vec<Vec<Edge>>,
}

impl Graph {
    /// Creates a new graph with `n` nodes and no edges.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    /// Adds a directed edge from node `u` to node `v` with weight `w`.
    pub fn add_edge(&mut self, u: NodeId, v: NodeId, w: Weight) {
        self.adj[u].push(Edge { to: v, w });
    }
}

#[derive(Clone, Copy, Debug)]
struct Params {
    k: usize, // depth bound / subtree threshold
    t: usize, // recursion coarse factor
}

impl Params {
    fn for_n(n: usize) -> Self {
        // Keep k,t >= 1; use ln for stability on small n.
        let ln = (n.max(2) as f64).ln();
        let k = ln.powf(1.0 / 3.0).floor().max(1.0) as usize;
        let t = ln.powf(2.0 / 3.0).floor().max(1.0) as usize;
        Self { k, t }
    }
}

/// Global state carried across calls.
#[derive(Clone, Debug)]
struct State {
    dist: Vec<Weight>,
    hops: Vec<usize>, // hop count for tie-breaks
    pred: Vec<Option<NodeId>>,
    complete: Vec<bool>, // whether finalized under current bound regime
}

impl State {
    fn new(n: usize) -> Self {
        Self {
            dist: vec![f64::INFINITY; n],
            hops: vec![usize::MAX; n],
            pred: vec![None; n],
            complete: vec![false; n],
        }
    }
}

/// Ordering key for frontier.
#[derive(Clone, Copy, Debug)]
struct Key {
    d: Weight,
    hop: usize,
    v: NodeId,
}
impl Eq for Key {}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.d == other.d && self.hop == other.hop && self.v == other.v
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap behavior over BinaryHeap
        match OrderedFloat(other.d).cmp(&OrderedFloat(self.d)) {
            Ordering::Equal => match other.hop.cmp(&self.hop) {
                Ordering::Equal => other.v.cmp(&self.v),
                o => o,
            },
            o => o,
        }
    }
}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Placeholder frontier with Pull/BatchPrepend semantics.
/// Internally BinaryHeap; interface matches the paper for future replacement.
struct PriorityFrontier {
    heap: BinaryHeap<Key>,
}

impl PriorityFrontier {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    fn insert(&mut self, k: Key) {
        self.heap.push(k);
    }

    fn batch_prepend(&mut self, batch: impl IntoIterator<Item = Key>) {
        for k in batch {
            self.heap.push(k);
        }
    }

    fn pull(&mut self, m: usize) -> (Vec<Key>, Weight) {
        let mut out = Vec::with_capacity(m);
        for _ in 0..m {
            if let Some(k) = self.heap.pop() {
                out.push(k);
            } else {
                break;
            }
        }

        // Approximate B as next smallest
        let next_b = self.heap.peek().map(|k| k.d).unwrap_or(f64::INFINITY);
        (out, next_b)
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

/// BaseCase(B, {x}): bounded Dijkstra with pop ≤ k and dist < B.
/// Returns (completed nodes, refined bound).
fn base_case(
    g: &Graph,
    st: &mut State,
    b_bound: Weight,
    src: NodeId,
    params: &Params,
) -> (Vec<NodeId>, Weight) {
    let mut heap = BinaryHeap::new();
    heap.push(Key {
        d: st.dist[src],
        hop: st.hops[src],
        v: src,
    });

    let mut popped = 0usize;
    let mut completed = Vec::new();
    let mut local_max = st.dist[src];

    while let Some(Key { d, hop, v }) = heap.pop() {
        if d > b_bound {
            break;
        }

        if st.complete[v] {
            continue;
        }

        st.complete[v] = true;
        completed.push(v);
        popped += 1;

        if d > local_max {
            local_max = d;
        }

        if popped > params.k {
            break;
        }

        for e in &g.adj[v] {
            let nd = d + e.w;
            if nd < b_bound && nd + 1e-15 < st.dist[e.to] {
                st.dist[e.to] = nd;
                st.hops[e.to] = hop.saturating_add(1);
                st.pred[e.to] = Some(v);
                heap.push(Key {
                    d: nd,
                    hop: st.hops[e.to],
                    v: e.to,
                });
            }
        }
    }

    (completed, local_max.min(b_bound))
}

/// FindPivots via shortest-path DAG approximation on W.
/// - Build W: nodes within k rounds from S following edges with nd < B.
/// - Build local DAG edges (u→v) only if:
///     (1) dist[u] + w ≈ dist[v]  (within EPS),
///     (2) forward in (dist,hop) order to ensure acyclicity,
///     (3) both under bound B.
/// - Compute forward-subtree size on DAG, clamped at k.
/// - Pick pivots = minimal roots with subtree_size ≥ k.
fn find_pivots(
    g: &Graph,
    st: &State,
    b_bound: Weight,
    sources: &BTreeSet<NodeId>,
    params: &Params,
) -> (BTreeSet<NodeId>, BTreeSet<NodeId>) {
    // 1) Collect W by k-round local exploration under bound B
    let mut in_q: VecDeque<NodeId> = sources.iter().copied().collect();
    let mut seen: BTreeSet<NodeId> = sources.clone();
    let mut rounds = 0usize;

    while rounds < params.k && !in_q.is_empty() {
        let mut next = VecDeque::new();
        while let Some(u) = in_q.pop_front() {
            // Only expand from nodes already within bound
            if !(st.dist[u] < b_bound) {
                continue;
            }
            for e in &g.adj[u] {
                let nd = st.dist[u] + e.w;
                if nd < b_bound - EPS {
                    if !seen.contains(&e.to) {
                        seen.insert(e.to);
                        next.push_back(e.to);
                    }
                }
            }
        }
        in_q = next;
        rounds += 1;
    }
    let w_set: BTreeSet<NodeId> = seen;
    if w_set.is_empty() {
        return (BTreeSet::new(), BTreeSet::new());
    }

    // 2) Build local DAG over W (forward in (dist,hop) to avoid cycles)
    // Index mapping for dense arrays
    let mut index: HashMap<NodeId, usize> = HashMap::with_capacity(w_set.len());
    for (i, &v) in w_set.iter().enumerate() {
        index.insert(v, i);
    }

    let mut nodes: Vec<NodeId> = w_set.iter().copied().collect();
    nodes.sort_by(
        |&a, &b| match OrderedFloat(st.dist[a]).cmp(&OrderedFloat(st.dist[b])) {
            Ordering::Equal => st.hops[a].cmp(&st.hops[b]).then_with(|| a.cmp(&b)),
            o => o,
        },
    );

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    // For each u in W, examine outgoing edges to v in W; add DAG edge if forward & equal-dist
    for &u in &nodes {
        let iu = index[&u];
        let du = st.dist[u];
        let hu = st.hops[u];
        if !(du.is_finite() && du < b_bound) {
            continue;
        }

        for e in &g.adj[u] {
            if let Some(&iv) = index.get(&e.to) {
                let dv = st.dist[e.to];
                if !(dv.is_finite() && dv < b_bound) {
                    continue;
                }
                let nd = du + e.w;

                // shortest-path equality (within EPS)
                if (nd - dv).abs() <= EPS {
                    // forward constraint to ensure acyclicity even with zero-weight edges
                    let forward =
                        (OrderedFloat(du), hu, u) < (OrderedFloat(dv), st.hops[e.to], e.to);
                    if forward {
                        children[iu].push(iv);
                        parents[iv].push(iu);
                    }
                }
            }
        }
    }

    // 3) Compute forward-subtree size (clamped at k) in reverse topological order
    // We'll process nodes in decreasing (dist,hop) so children first.
    let mut order = nodes.clone();
    order.sort_by(|&a, &b| {
        match OrderedFloat(st.dist[b]).cmp(&OrderedFloat(st.dist[a])) {
            // reverse
            Ordering::Equal => st.hops[b].cmp(&st.hops[a]).then_with(|| b.cmp(&a)),
            o => o,
        }
    });

    let mut size: Vec<usize> = vec![0; nodes.len()];
    for v in order {
        let iv = index[&v];
        let mut acc = 1usize;
        for &cw in &children[iv] {
            // clamp to params.k to keep it small
            acc = acc.saturating_add(size[cw]);
            if acc >= params.k {
                acc = params.k;
                break;
            }
        }
        size[iv] = acc;
    }

    // 4) Pick minimal roots: size≥k and no parent with size≥k
    let mut pivots = BTreeSet::new();
    for &v in &nodes {
        let iv = index[&v];
        if size[iv] >= params.k {
            let has_big_parent = parents[iv].iter().any(|&p| size[p] >= params.k);
            if !has_big_parent {
                pivots.insert(v);
            }
        }
    }

    // Fallback: if pivots empty (e.g., W too small), ensure S survives
    if pivots.is_empty() {
        // Choose at least one representative near the "front" of W.
        // Prefer nodes with smallest (dist,hop) among W ∩ S, else among W.
        if let Some(&best_s) = nodes
            .iter()
            .filter(|&&v| sources.contains(&v))
            .min_by(
                |&&a, &&b| match OrderedFloat(st.dist[a]).cmp(&OrderedFloat(st.dist[b])) {
                    Ordering::Equal => st.hops[a].cmp(&st.hops[b]).then_with(|| a.cmp(&b)),
                    o => o,
                },
            )
        {
            pivots.insert(best_s);
        } else {
            pivots.insert(nodes[0]); // safe: W non-empty
        }
    }

    (w_set, pivots)
}

/// bounded multi-source SSSP with partial exits.
/// Returns (U: completed set, B': refined bound).
fn bmssp(
    g: &Graph,
    st: &mut State,
    level: usize,
    b_bound: Weight,
    sources: &BTreeSet<NodeId>,
    params: &Params,
) -> (BTreeSet<NodeId>, Weight) {
    if sources.is_empty() {
        return (BTreeSet::new(), b_bound);
    }

    // Base level: run BaseCase per source and merge (prevents infinite recursion).
    if level == 0 {
        let mut completed_all = BTreeSet::new();
        let mut best_b = b_bound;
        for &s in sources {
            let (u0, bprime) = base_case(g, st, b_bound, s, params);
            completed_all.extend(u0);
            best_b = best_b.min(bprime);
        }
        return (completed_all, best_b);
    }

    // 1) Improved FindPivots (M1)
    let (_w, pivots) = find_pivots(g, st, b_bound, sources, params);

    // 2) Initialize frontier
    let mut front = PriorityFrontier::new();
    for &x in &pivots {
        front.insert(Key {
            d: st.dist[x],
            hop: st.hops[x],
            v: x,
        });
    }

    // 3) Process batches
    let mut completed: BTreeSet<NodeId> = BTreeSet::new();
    let mut best_b = b_bound;

    // threshold: k * 2^{l·t} with saturation
    let shift: u32 = (level as u32).saturating_mul(params.t as u32);
    let pow2: usize = 2usize.saturating_pow(shift);
    let thresh: usize = params.k.saturating_mul(pow2);

    while !front.is_empty() {
        let m = 8usize; // placeholder batch size for heap-backed frontier
        let (batch, next_b) = front.pull(m);
        let batch_set: BTreeSet<NodeId> = batch.iter().map(|k| k.v).collect();

        let (u_i, b_i_prime) = if !batch_set.is_empty() {
            if level > 0 {
                bmssp(g, st, level - 1, next_b.min(b_bound), &batch_set, params)
            } else {
                // Defensive: should not happen because of early return, but keep safe path.
                let mut all = BTreeSet::new();
                let mut bbest = b_bound;
                for &s in &batch_set {
                    let (u0, b0) = base_case(g, st, next_b.min(b_bound), s, params);
                    all.extend(u0);
                    bbest = bbest.min(b0);
                }
                (all, bbest)
            }
        } else {
            (BTreeSet::new(), next_b.min(b_bound))
        };
        best_b = best_b.min(b_i_prime);

        // Relax outgoing edges of completed nodes, collecting new candidates
        let mut to_prepend = Vec::new();
        for &u in &u_i {
            if !st.complete[u] {
                st.complete[u] = true;
                completed.insert(u);
            }
            let du = st.dist[u];
            for e in &g.adj[u] {
                let nd = du + e.w;
                if nd < b_bound && nd + 1e-15 < st.dist[e.to] {
                    st.dist[e.to] = nd;
                    st.hops[e.to] = st.hops[u].saturating_add(1);
                    st.pred[e.to] = Some(u);
                    to_prepend.push(Key {
                        d: nd,
                        hop: st.hops[e.to],
                        v: e.to,
                    });
                } else if nd < b_bound && (nd - st.dist[e.to]).abs() <= 1e-15 {
                    // Keep equal-distance reinsertion (helps path-lex stability).
                    to_prepend.push(Key {
                        d: nd,
                        hop: st.hops[u].saturating_add(1),
                        v: e.to,
                    });
                }
            }
        }
        front.batch_prepend(to_prepend);

        if completed.len() >= thresh && thresh > 0 {
            break;
        }
    }

    (completed, best_b)
}

/// progressively refines bound B while advancing the frontier S.
pub fn directed_sssp(g: &Graph, source: NodeId) -> Vec<Weight> {
    let n = g.n;
    let params = Params::for_n(n);
    let max_levels = ((n as f64).ln() / (params.t as f64).max(1.0)).ceil() as usize;

    let mut st = State::new(n);
    st.dist[source] = 0.0;
    st.hops[source] = 0;

    let mut S = BTreeSet::new();
    S.insert(source);
    let mut B = f64::INFINITY;

    for l in (0..=max_levels).rev() {
        let (_u, bprime) = bmssp(g, &mut st, l, B, &S, &params);
        B = bprime;

        // New frontier: nodes with dist < B but not yet completed
        S = (0..n)
            .filter(|&v| st.dist[v].is_finite() && st.dist[v] < B && !st.complete[v])
            .collect();

        if S.is_empty() {
            break;
        }
    }
    st.dist
}

/// Reference Dijkstra implementation for correctness testing.
#[allow(dead_code)]
fn dijkstra_ref(g: &Graph, s: NodeId) -> Vec<Weight> {
    let mut dist = vec![f64::INFINITY; g.n];
    let mut hops = vec![usize::MAX; g.n];
    let mut heap = BinaryHeap::new();
    dist[s] = 0.0;
    hops[s] = 0;
    heap.push(Key {
        d: 0.0,
        hop: 0,
        v: s,
    });
    while let Some(Key { d, hop, v }) = heap.pop() {
        if d > dist[v] {
            continue;
        }
        for e in &g.adj[v] {
            let nd = d + e.w;
            if nd + 1e-15 < dist[e.to] {
                dist[e.to] = nd;
                hops[e.to] = hop + 1;
                heap.push(Key {
                    d: nd,
                    hop: hop + 1,
                    v: e.to,
                });
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn tiny_dag_matches_dijkstra() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 3, 4.0);
        g.add_edge(3, 4, 1.0);
        g.add_edge(2, 4, 1.0);

        let bm = directed_sssp(&g, 0);
        let dj = dijkstra_ref(&g, 0);

        for v in 0..g.n {
            assert!(
                (bm[v] - dj[v]).abs() < 1e-9,
                "v={}: {} vs {}",
                v,
                bm[v],
                dj[v]
            );
        }
    }

    #[test]
    fn random_sparse_graph_consistency() {
        let n = 200;
        let mut g = Graph::new(n);
        let mut rng = StdRng::seed_from_u64(42);
        for u in 0..n {
            // avg out-degree ~ 4
            for _ in 0..4 {
                let v = rng.random_range(0..n);
                if u != v {
                    let w = rng.random_range(1..10) as f64;
                    g.add_edge(u, v, w);
                }
            }
        }

        let s = 0usize;
        let bm = directed_sssp(&g, s);
        let dj = dijkstra_ref(&g, s);

        for v in 0..n {
            assert!(
                (bm[v].is_infinite() && dj[v].is_infinite()) || (bm[v] - dj[v]).abs() < 1e-7,
                "v={}: {} vs {}",
                v,
                bm[v],
                dj[v]
            );
        }
    }
}
