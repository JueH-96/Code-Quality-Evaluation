import sys
from itertools import combinations

# Since N is small, the default recursion limit is sufficient.

# DSU (Disjoint Set Union) class to efficiently check for cycles.
# Also known as the Union-Find data structure.
class DSU:
    """
    A Disjoint Set Union (DSU) data structure with path compression.
    It is used to efficiently track a partition of a set into disjoint subsets.
    """
    def __init__(self, n):
        # The problem uses 1-indexed vertices (1 to N), so the parent array
        # needs to be of size n+1 to accommodate vertex N.
        # Initially, each vertex is in its own set.
        self.parent = list(range(n + 1))

    def find(self, i):
        """
        Finds the representative (or root) of the set containing element i.
        Uses path compression for optimization: during the find operation, it
        makes all nodes on the path point directly to the root.
        """
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        Merges the sets containing elements i and j by making the root of one
        point to the root of the other.
        Returns True if they were in different sets (a merge happened),
        and False if they were already in the same set. A False return value
        indicates that adding an edge between i and j would form a cycle in
        the graph built so far.
        """
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False

# Read the number of vertices, edges, and the modulus K.
N, M, K = map(int, sys.stdin.readline().split())

# Read all M edges into a list.
edges = []
for _ in range(M):
    u, v, w = map(int, sys.stdin.readline().split())
    edges.append((u, v, w))

# Initialize the minimum cost found so far to a value larger than any possible cost.
# The cost is a value modulo K, so it will be in the range [0, K-1].
# float('inf') is a safe choice.
min_cost = float('inf')

# A spanning tree on a graph with N vertices must have exactly N-1 edges.
num_edges_in_st = N - 1

# The core of the solution: The number of vertices N is very small (<= 8).
# This allows us to check every possible candidate for a spanning tree.
# We generate all combinations of N-1 edges from the M available edges.
for edge_subset in combinations(edges, num_edges_in_st):
    
    # For each combination, we verify that it forms a valid spanning tree.
    # A property of graphs states that an undirected graph with N vertices and
    # N-1 edges is a tree if and only if it is acyclic.
    # We use a DSU data structure to detect cycles efficiently.
    dsu = DSU(N)
    is_acyclic = True
    
    for u, v, _ in edge_subset:
        # The union operation returns False if u and v are already connected.
        # In that case, adding the edge (u, v) would create a cycle.
        if not dsu.union(u, v):
            is_acyclic = False
            break
    
    # If the loop completed without finding any cycles, the N-1 edges form a spanning tree.
    if is_acyclic:
        # Calculate the total weight of the edges in this spanning tree.
        # Python's integers handle arbitrarily large numbers, so there is no
        # risk of overflow even with large K and w_i.
        current_sum_of_weights = sum(w for _, _, w in edge_subset)
        
        # The cost is defined as the total weight modulo K.
        cost = current_sum_of_weights % K
        
        # Keep track of the minimum cost found across all valid spanning trees.
        min_cost = min(min_cost, cost)

# After checking all combinations, print the overall minimum cost.
print(min_cost)