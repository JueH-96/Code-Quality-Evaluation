import sys
from collections import deque

# Set recursion depth for DFS in max flow
sys.setrecursionlimit(2000) # N * D can be up to 30 * 30 = 900, plus S and T. Depth could be around N * D. Set a reasonable limit.
INF = 10**9 # A large number for infinite capacity

# Dinic's Algorithm
class Edge:
    def __init__(self, to, capacity, rev):
        self.to = to
        self.capacity = capacity
        self.rev = rev

def add_edge(adj, edges, fr, to, capacity):
    adj[fr].append(len(edges))
    edges.append(Edge(to, capacity, len(edges) + 1))
    adj[to].append(len(edges))
    edges.append(Edge(fr, 0, len(edges) - 1)) # Residual edge

def bfs(s, t, n_nodes, adj, edges, level):
    """Builds the level graph"""
    for i in range(n_nodes):
        level[i] = -1
    level[s] = 0
    q = deque([s])
    while q:
        v = q.popleft()
        for id in adj[v]:
            e = edges[id]
            if e.capacity > 0 and level[e.to] < 0:
                level[e.to] = level[v] + 1
                q.append(e.to)
    return level[t] != -1

def dfs(v, t, f, adj, edges, level, iter_list):
    """Finds augmenting path in the level graph"""
    if v == t:
        return f
    # Use iter_list to track progress in adjacency list for current DFS path
    # This is an optimization for Dinic
    while iter_list[v] < len(adj[v]):
        id = adj[v][iter_list[v]]
        e = edges[id]
        # Check if edge has capacity, goes to the next level, and is part of the level graph
        if e.capacity > 0 and level[v] < level[e.to]:
            d = dfs(e.to, t, min(f, e.capacity), adj, edges, level, iter_list)
            if d > 0:
                e.capacity -= d
                edges[e.rev].capacity += d
                return d
        iter_list[v] += 1 # Move to next edge in adjacency list for node v
    return 0

def max_flow(s, t, n_nodes, adj, edges):
    """Computes the maximum flow from s to t"""
    flow = 0
    while True: # Loop until no more augmenting paths can be found
        level = [-1] * n_nodes
        if not bfs(s, t, n_nodes, adj, edges, level):
            break # No more augmenting paths exist

        # Find all augmenting paths in the current level graph
        iter_list = [0] * n_nodes # Reset iter_list for the new level graph
        # Find paths from s to t in the current level graph until no more are found
        f = dfs(s, t, float('inf'), adj, edges, level, iter_list)
        while f > 0:
             flow += f
             # After finding one path, try to find another path from s in the *same* level graph
             # The iter_list is not reset here, allowing subsequent DFS calls from s
             # to continue exploring from where previous searches left off for each node v.
             # The standard implementation restarts DFS from s without resetting iter_list
             # fully, but rather relies on the state being kept for each node.
             # Let's reset iter_list from s only.
             iter_list[s] = 0 # Reset progress from source for the next path search
             f = dfs(s, t, float('inf'), adj, edges, level, iter_list)

    return flow

# Check function using Max Flow
def check(D, N, M, K, original_edges):
    """
    Checks if it's possible to make the shortest distance from 1 to N
    at least D by setting at most K edges to weight 1.
    This is true if the minimum number of edges required to make the
    shortest path >= D is <= K. The minimum number of edges required
    is found by the min cut value in a specific layered graph.
    """
    if D == 0:
        return True # Shortest distance is always >= 0

    # Layered graph nodes v_i for v=1..N, i=0..D-1
    # These nodes represent reaching vertex v having used exactly i edges of weight 1.
    # Total N * D nodes, indexed 0 to N*D - 1
    # Source node S = N*D, Sink node T = N*D + 1
    num_flow_nodes = N * D + 2
    s = N * D
    t = N * D + 1

    adj = [[] for _ in range(num_flow_nodes)]
    edges = []

    # Map (v, i) to node index (v is 1-based, i is 0-based layer)
    node_idx = lambda v, i: (v-1)*D + i

    # Edge from Source S to 1_0 with infinite capacity
    # Represents starting at vertex 1 with 0 edges of weight 1 used so far.
    add_edge(adj, edges, s, node_idx(1, 0), INF)

    # Edge from N_i to Sink T with infinite capacity for i = 0..D-1
    # Reaching the sink means reaching vertex N having used < D edges of weight 1.
    for i in range(D):
        add_edge(adj, edges, node_idx(N, i), t, INF)

    # Edges based on original graph edges (u, v)
    for u, v in original_edges:
        for i in range(D):
            # u_i -> v_i (weight 0, infinite capacity)
            # Using original edge (u,v) as a 0-edge transition from layer i to i.
            add_edge(adj, edges, node_idx(u, i), node_idx(v, i), INF)

            # u_i -> v_{i+1} (weight 1, capacity 1)
            # Using original edge (u,v) as a 1-edge transition from layer i to i+1.
            # This edge exists only if the target layer i+1 is less than D.
            if i + 1 < D:
                add_edge(adj, edges, node_idx(u, i), node_idx(v, i+1), 1)

    # The minimum S-T cut in this graph represents the minimum number of capacity-1 edges
    # that must be cut to disconnect S from T. This is equivalent to the minimum number
    # of original edges that must be selected to have weight 1 such that there is no
    # path from 1 to N with fewer than D edges of weight 1.
    min_cut_value = max_flow(s, t, num_flow_nodes, adj, edges)

    # If the minimum number of edges required is <= K, then it is possible.
    return min_cut_value <= K

# Read input
N, M, K = map(int, sys.stdin.readline().split())
original_edges = []
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    original_edges.append((u, v))

# Binary search for the maximum possible shortest distance
# The shortest distance can be between 0 and N-1 (for simple path)
# or up to M (if multi-edges allow many 1-edges on a path).
# A safe upper bound for the shortest distance is N or M or K.
# The number of 1-edges on a shortest path in a 0-1 graph on N vertices is at most N-1.
# But the definition of shortest path is minimum sum of weights. The path itself is simple.
# The max number of edges on a simple path is N-1. The max number of 1s is at most N-1.
# Using N as the upper bound for the possible shortest distance seems reasonable.
# Binary search range [0, N+1].
low = 0
high = N + 1 # Search space is [0, 1, ..., N]

while high - low > 1:
    mid = low + (high - low) // 2
    if check(mid, N, M, K, original_edges):
        # It is possible to achieve a shortest distance of at least 'mid'
        low = mid
    else:
        # It is not possible to achieve a shortest distance of at least 'mid'
        high = mid

# The maximum possible shortest distance is the largest 'mid' for which check(mid) was True
print(low)