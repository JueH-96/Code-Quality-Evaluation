import sys
import math

# Helper for DSU
class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

# BFS approach for path sum reconstruction in a tree
def get_path_sum_in_tree_bfs_recon(adj, start, end):
    q = [start]
    visited = {start}
    parent_edge = {start: None} # {node: (parent_node, edge_weight)}

    while q:
        curr = q.pop(0)
        if curr == end:
            break

        # Ensure node exists in adj before iterating
        if curr not in adj:
             continue # Should not happen in a connected tree

        for neighbor, weight in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
                parent_edge[neighbor] = (curr, weight)
    
    # Reconstruct path sum
    current = end
    total_path_sum = 0
    while parent_edge[current] is not None:
        prev, weight = parent_edge[current]
        total_path_sum += weight
        current = prev
        
    return total_path_sum

# Read input
N, M, K = map(int, sys.stdin.readline().split())
edges = []
for _ in range(M):
    u, v, w = map(int, sys.stdin.readline().split())
    edges.append((u, v, w))

# 1. Find a spanning tree T0 (using Kruskal's logic) and its sum S0
# We don't need to sort edges for just finding *a* spanning tree, any order works.
dsu = DSU(N)
s0 = 0
tree_edges = []

edges_count = 0
# Iterate through edges to build a spanning tree
# Using the order from input is fine
for u, v, w in edges:
    if dsu.union(u, v):
        s0 += w
        tree_edges.append((u, v, w))
        edges_count += 1
        if edges_count == N - 1:
            break # Found a spanning tree

# Identify non-tree edges
# We need to iterate through all original edges to find those not in T0
tree_edge_set = set()
for u, v, w in tree_edges:
    # Canonical representation for edge (u,v)
    tree_edge_set.add(tuple(sorted((u, v))))

non_tree_edges = []
for u, v, w in edges:
    if tuple(sorted((u, v))) not in tree_edge_set:
        non_tree_edges.append((u, v, w))


# 2. Build adjacency list for T0
adj_t0 = {i: [] for i in range(1, N + 1)}
for u, v, w in tree_edges:
    adj_t0[u].append((v, w))
    adj_t0[v].append((u, w))

# 3. Calculate fundamental cycle sums relative to T0
# For each non-tree edge, add it to T0 to form a cycle. Calculate the sum of weights in this cycle.
cycle_sums = []
for u, v, w_uv in non_tree_edges:
    # Find path in T0 between u and v
    path_sum = get_path_sum_in_tree_bfs_recon(adj_t0, u, v)
    
    # The sum of weights in the cycle formed by edge (u,v) and the path in T0
    # is w_uv + path_sum
    cycle_sum = w_uv + path_sum
    cycle_sums.append(cycle_sum)

# 4. Compute all achievable sums modulo K
# The set of achievable sums modulo K is { (S0 + sum_{i in I} C'_i) mod K } for I subset {1..p}
# where C'_i are the fundamental cycle sums calculated above.
# Use a set to store unique achievable costs modulo K.
achievable_costs = {s0 % K}

# Iterate through each fundamental cycle sum and update the set of achievable costs
for cycle_sum in cycle_sums:
    new_costs = set()
    for cost in achievable_costs:
        new_costs.add((cost + cycle_sum) % K)
    achievable_costs.update(new_costs)

# 5. Find the minimum cost
min_cost = min(achievable_costs)

# 6. Print the minimum cost
print(min_cost)