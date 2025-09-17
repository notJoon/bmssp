use core::f64;
use std::{
    cmp::Ordering,
    collections::{BTreeSet, BinaryHeap, HashSet, VecDeque},
};

use ordered_float::OrderedFloat;

type NodeId = usize;
type Weight = f64;

#[derive(Debug, Clone)]
pub struct Edge {
    pub to: NodeId,
    pub w: Weight,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub n: usize,
    pub adj: Vec<Vec<Edge>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    pub fn add_edge(&mut self, u: NodeId, v: NodeId, w: Weight) {
        self.adj[u].push(Edge { to: v, w });
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub k: usize, // depth bound for limited relax & base case
    pub t: usize, // controls recursion depth: levels ~ (log n)/t
}

impl Params {
    fn for_n(n: usize) -> Self {
        // Safe guards: ensure k,t >= 1 for tiny graphs
        let ln = (n.max(2) as f64).ln();
        let k = ln.powf(1.0 / 3.0).floor().max(1.0) as usize;
        let t = ln.powf(2.0 / 3.0).floor().max(1.0) as usize;
        Self { k, t }
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub dist: Vec<Weight>,
    pub hops: Vec<usize>, // for tie-breaking (track hop count)
    pub pred: Vec<Option<NodeId>>,
    pub complete: Vec<bool>, // "completed" (distance finalized under current bound)
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

/// Key used in frontier ordering.
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

/// Placeholder frontier with Pull, BatchPrepend semantics.
struct PriorityFrontier {
    // TODO: Internally uses `BinaryHeap` for now.
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

/// k-round limited relax (Bellman-Ford style) from multi-source S,
/// collecting nodes within distance < B. Returns (W, P) where
/// P is a reduced pivot set (currently: sample 1 out of k or S when small).
/// Algoroth 1
fn find_pivots(
    g: &Graph,
    st: &State,
    b_bound: Weight,
    sources: &BTreeSet<NodeId>,
    params: &Params,
) -> (BTreeSet<NodeId>, BTreeSet<NodeId>) {
    let mut in_q: VecDeque<NodeId> = sources.iter().copied().collect();
    let mut seen = HashSet::new();
    for &s in sources {
        seen.insert(s);
    }
    let mut rounds = 0usize;

    while rounds < params.k && !in_q.is_empty() {
        let mut next = VecDeque::new();
        while let Some(u) = in_q.pop_front() {
            for e in &g.adj[u] {
                // Use current distances
                // only follow edges that stay under bound
                let nd = st.dist[u] + e.w;
                if nd < b_bound && !seen.contains(&e.to) {
                    // NOTE: we don't update global state here; this is just a reachability lens
                    seen.insert(e.to);
                    next.push_back(e.to);
                }
            }
        }
        in_q = next;
        rounds += 1;
    }
    // W = nodes touched (including sources)
    let mut w: BTreeSet<NodeId> = seen.into_iter().collect();
    w.extend(sources.iter().copied());

    // Very simple pivot reduction: pick ~|W|/k nodes by sampling layers.
    // Later: replace with "roots of size ≥ k shortest-path trees".
    let mut p = BTreeSet::new();
    if w.len() <= params.k {
        p = sources.clone();
    } else {
        for (i, v) in w.iter().copied().enumerate() {
            if i % params.k == 0 {
                p.insert(v);
            }
        }
    }
    (w, p)
}

/// Algorithm #2
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
        // This is the moment we "finalize" v under current bound
        st.complete[v] = true;
        completed.push(v);
        popped += 1;
        if d > local_max {
            local_max = d;
        }
        if popped > params.k {
            // depth-limited; stop early
            break;
        }
        // relax outgoing edges
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
    // refined bound: max distance we actually finalized in this round
    let b_prime = local_max.min(b_bound);
    (completed, b_prime)
}

/// bounded multi-source SSSP
/// Algorith #3
pub fn bmssp(
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
    // When level == 0, regardless of the number of sources (1 or more),
    // execute BaseCase for each source and merge the results
    if level == 0 {
        let mut completed_all = BTreeSet::new();
        let mut best_b = b_bound;

        // Process each source individually
        for &s in sources {
            let (u0, bprime) = base_case(g, st, b_bound, s, params);
            completed_all.extend(u0); // merge completed nodes from this source
            best_b = best_b.min(bprime); // keep track of the best bound found
        }
        return (completed_all, best_b);
    }

    // 1) Find pivots & local window W
    let (_w, pivots) = find_pivots(g, st, b_bound, sources, params);

    // 2) Initialize frontier with pivot keys
    let mut front = PriorityFrontier::new();
    for &x in &pivots {
        front.insert(Key {
            d: st.dist[x],
            hop: st.hops[x],
            v: x,
        });
    }

    // 3) Main loop: pull batches, solve recursively, relax out-neighbors, batch-prepend
    let mut completed: BTreeSet<NodeId> = BTreeSet::new();
    let mut best_b = b_bound;

    // Partial exit threshold: |U| ≥ k * 2^{l*t}
    let shift: u32 = (level as u32).saturating_mul(params.t as u32);
    let pow2: usize = 2usize.saturating_pow(shift);
    let thresh: usize = params.k.saturating_mul(pow2);

    while !front.is_empty() {
        // Pull a small batch; heuristic M = max(1, |front|/k) but with heap we don't know |front|
        let m = 8usize; // placeholder batch size
        let (batch, next_b) = front.pull(m);
        let batch_set: BTreeSet<NodeId> = batch.iter().map(|k| k.v).collect();

        // Recursive solve under upper bound next_b
        let (u_i, b_i_prime) = if !batch_set.is_empty() {
            if level > 0 {
                bmssp(g, st, level - 1, next_b.min(b_bound), &batch_set, params)
            } else {
                // defensive: safely handled it with `base_case`
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
        // Mark completed and relax their outgoing edges (producing new candidates)
        let mut to_prepend = Vec::new();

        for &u in &u_i {
            if !st.complete[u] {
                st.complete[u] = true;
                completed.insert(u);
            }
            for e in &g.adj[u] {
                let nd = st.dist[u] + e.w;
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
                    // Equal-distance tie: still consider enqueue to support path-lex ordering later
                    to_prepend.push(Key {
                        d: nd,
                        hop: st.hops[u].saturating_add(1),
                        v: e.to,
                    });
                }
            }
        }
        front.batch_prepend(to_prepend);

        // keep U bounded
        if completed.len() >= thresh && thresh > 0 {
            break;
        }
    }

    (completed, best_b)
}

/// Public driver: computes all-pairs from a single source using BMSSP layering.
/// For simplicity, we grow the bound B progressively using a min frontier.
pub fn directed_sssp(g: &Graph, source: NodeId) -> Vec<Weight> {
    let n = g.n;
    let params = Params::for_n(n);
    let max_levels = ((n as f64).ln() / (params.t as f64).max(1.0)).ceil() as usize;

    let mut st = State::new(n);
    st.dist[source] = 0.0;
    st.hops[source] = 0;

    // Initial frontier & bound
    let mut big_s = BTreeSet::new();
    big_s.insert(source);
    let mut big_b: f64 = f64::INFINITY;

    // Iterate levels from coarse to fine bound ranges
    for l in (0..=max_levels).rev() {
        let (_u, bprime) = bmssp(g, &mut st, l, big_b, &big_s, &params);
        big_b = bprime;
        // Expand S to boundary layer: nodes with dist < B but not completed
        big_s = (0..n)
            .filter(|&v| st.dist[v].is_finite() && st.dist[v] < big_b && !st.complete[v])
            .collect();
        if big_s.is_empty() {
            break;
        }
    }
    st.dist
}

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
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
            // allow small numerical slop
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
