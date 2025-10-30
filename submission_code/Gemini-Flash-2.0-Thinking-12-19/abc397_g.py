import sys
from collections import deque

# Dinic's algorithm
class Edge:
    def __init__(self, to, capacity):
        self.to = to
        self.capacity = capacity
        self.rev = -1

def add_edge(adj, u, v, capacity):
    # Add edge u -> v with capacity, and v -> u with residual capacity 0
    adj[u].append(Edge(v, capacity))
    adj[v].append(Edge(u, 0))
    # Set residual edge index
    adj[u][-1].rev = len(adj[v]) - 1
    adj[v][-1].rev = len(adj[u]) - 1

def bfs(adj, s, t, level):
    """Computes levels for Dinic's algorithm."""
    # Initialize levels
    for i in range(len(level)):
        level[i] = -1
    level[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for edge in adj[u]:
            # If edge has remaining capacity and endpoint is not leveled yet
            if edge.capacity > 0 and level[edge.to] < 0:
                level[edge.to] = level[u] + 1
                q.append(edge.to)
    # Return True if sink is reachable, False otherwise
    return level[t] != -1

def dfs(adj, s, t, level, iter_edge, flow):
    """Finds augmenting paths in the level graph using DFS."""
    if s == t:
        return flow # Reached sink, return found flow
    
    # Iterate through adjacent edges starting from the last visited edge
    while iter_edge[s] < len(adj[s]):
        edge = adj[s][iter_edge[s]]
        # If edge has capacity, goes to next level, and target is leveled
        if edge.capacity > 0 and level[s] < level[edge.to]:
            # Recursively find flow from the next node
            d = dfs(adj, edge.to, t, level, iter_edge, min(flow, edge.capacity))
            if d > 0:
                # Augment flow along the path
                edge.capacity -= d
                adj[edge.to][edge.rev].capacity += d
                return d # Return found flow
        iter_edge[s] += 1 # Move to the next edge
    return 0 # No augmenting path found from this node

def max_flow(adj, s, t):
    """Computes maximum flow from s to t using Dinic's algorithm."""
    flow = 0
    level = [-1] * len(adj)
    iter_edge = [0] * len(adj)

    while bfs(adj, s, t, level):
        # While there is a path from s to t in the level graph
        iter_edge = [0] * len(adj) # Reset edge iterators for each BFS phase
        while True:
            # Find augmenting paths using DFS
            f = dfs(adj, s, t, level, iter_edge, float('inf'))
            if f == 0:
                break # No more augmenting paths in this level graph
            flow += f # Add found flow to total
    return flow

# Check if shortest path >= X is possible using <= K weight-1 edges
def can_achieve_distance(N, M, K, edges, X):
    """
    Checks if it's possible to assign K edges weight 1 such that the shortest
    distance from 1 to N is at least X. This is done by checking if the minimum
    number of edges that *must* be assigned weight 1 to disconnect all paths
    from 1 to N with < X weight-1 edges is <= K. This minimum number is given
    by the min-cut in a constructed flow network.
    """
    # If target distance is 0, it's always achievable since N is reachable from 1.
    # Min cost is 0 (no edges weight 1), and 0 <= K.
    if X == 0:
        return True
        
    # Build flow network
    # Nodes: (u, k) for u in [1, N], k in [0, X]
    # State (u, k) means vertex u reached with exactly k weight-1 edges.
    # Plus source S and sink T.
    # Total nodes = N * (X + 1) + 2
    num_nodes = N * (X + 1) + 2
    S = 0
    T = num_nodes - 1
    adj = [[] for _ in range(num_nodes)]
    
    # Map (u, k) to node index
    # (u, k) -> 1 + (u-1) * (X + 1) + k
    def get_node_idx(u, k, X_val):
        # u is 1-indexed, k is 0-indexed
        return 1 + (u - 1) * (X_val + 1) + k

    # Source S to initial state (1, 0)
    # Capacity infinity: represents starting at node 1 with 0 weight-1 edges is free.
    add_edge(adj, S, get_node_idx(1, 0, X), float('inf'))
    
    # Add edges corresponding to original graph edges (u_j, v_j)
    for u_j, v_j in edges:
        # Option 1: Assign original edge (u_j, v_j) weight 1
        # This costs 1 unit in the min-cut (capacity 1 edge)
        # State transition: (u_j, k) -> (v_j, k+1)
        # This transition is possible only if current weight-1 count k is less than X
        for k in range(X): # k goes from 0 to X-1. Reaches state k+1 from 1 to X.
             add_edge(adj, get_node_idx(u_j, k, X), get_node_idx(v_j, k + 1, X), 1)
        
        # Option 2: Assign original edge (u_j, v_j) weight 0
        # This costs infinity in the min-cut (capacity infinity edge)
        # State transition: (u_j, k) -> (v_j, k)
        # This transition is possible for any current weight-1 count k up to X.
        for k in range(X + 1): # k goes from 0 to X
             add_edge(adj, get_node_idx(u_j, k, X), get_node_idx(v_j, k, X), float('inf'))

    # Sink connections: (N, k) for k in [0, X-1] to T
    # Reaching node N with fewer than X weight-1 edges means success for the flow.
    # These nodes are connected to the sink with capacity infinity.
    # The min-cut aims to separate S from T. To separate, it must cut paths
    # that reach these sink-connected nodes. Capacity 1 edges on these paths
    # represent the edges we choose to assign weight 1.
    for k in range(X): # k from 0 to X-1
        add_edge(adj, get_node_idx(N, k, X), T, float('inf'))
        
    # The min-cut capacity in this network equals the minimum number of original
    # edges that must be assigned weight 1 to ensure that any path from 1 to N
    # in the resulting graph uses at least X weight-1 edges.
    min_edges_needed = max_flow(adj, S, T)
    
    # If the minimum number of edges required is less than or equal to K (our budget),
    # then it is possible to achieve a shortest distance of at least X.
    return min_edges_needed <= K

# Read input
N, M, K = map(int, sys.stdin.readline().split())
edges = []
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    edges.append((u, v))

# Binary search for the maximum possible shortest distance X.
# The shortest distance can range from 0 up to N (as a loose upper bound, N-1 for simple path).
# If N=2 and edge 1->2 exists and K >= 1, shortest path can be 1. Max N.
# Search range [L, R). L is potential answer, R is upper bound.
L = 0 # Minimum possible shortest distance
R = N + 1 # Maximum possible shortest distance + 1

while R - L > 1:
    mid = L + (R - L) // 2 # Check if shortest distance 'mid' is achievable
    if can_achieve_distance(N, M, K, edges, mid):
        # If distance 'mid' is achievable, try for a larger distance
        L = mid
    else:
        # If distance 'mid' is not achievable, the max possible distance must be less than 'mid'
        R = mid

# After binary search, L holds the maximum possible shortest distance.
print(L)