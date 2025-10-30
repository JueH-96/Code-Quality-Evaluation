import sys

# DSU (Disjoint Set Union) structure with union by size and path compression
class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i.
        # If parent[i] == i, then i is the root of its set.
        # Vertices are 1-indexed, so we use n+1 size and ignore index 0.
        self.parent = list(range(n + 1)) 
        # size[i] stores the size of the tree rooted at i (if i is a root).
        self.size = [1] * (n + 1)
        # Keeps track of the total number of disjoint sets.
        self.num_components = n

    def find(self, i):
        # Find the root of the set containing i.
        # Implements path compression: makes every node on the path point directly to the root.
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Union the sets containing i and j.
        # Returns True if a union was performed (i.e., i and j were in different sets), False otherwise.
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by size: attach the smaller tree to the root of the larger tree.
            # This keeps the tree flatter, improving performance for future find operations.
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i # Swap to ensure root_i is the larger or equal sized tree
            
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.num_components -= 1 # One less component after merging
            return True
        return False

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # List to store all candidate edges for Kruskal's algorithm.
    # Each edge is stored as (weight, u, v).
    edges = []

    for _ in range(M):
        # Read K_i and C_i from the first line of the operation block
        line1_parts = list(map(int, sys.stdin.readline().split()))
        K_i = line1_parts[0]
        C_i = line1_parts[1]
        
        # Read the K_i vertices A_{i,1}, ..., A_{i,K_i}
        # These are guaranteed to be sorted: A_{i,1} < A_{i,2} < ...
        A_i = list(map(int, sys.stdin.readline().split()))
        
        # For each operation, instead of adding K_i * (K_i - 1) / 2 edges (a clique),
        # we add K_i - 1 edges to form a "star" centered at A_i[0] (which is A_{i,1}).
        # This is sufficient for Kruskal's algorithm to find the MST correctly.
        # If any two vertices u, v in S_i are not connected by cheaper edges,
        # they will eventually be connected via A_i[0] with cost C_i, or directly
        # by an actual edge (u,v) of cost C_i if that turns out to be part of the MST.
        # The star configuration ensures that all vertices in S_i *can* be connected
        # using edges of cost C_i.
        
        first_vertex_in_set = A_i[0] # This is A_{i,1}
        for j in range(1, K_i): # Iterate from the second vertex (A_{i,2}) to the last (A_{i,K_i})
            edges.append((C_i, first_vertex_in_set, A_i[j]))

    # Sort all candidate edges by weight in non-decreasing order.
    # This is a core step of Kruskal's algorithm.
    edges.sort()

    # Initialize the Disjoint Set Union (DSU) data structure.
    # Each vertex (1 to N) starts in its own component.
    dsu = DSU(N)
    mst_weight = 0

    # Iterate through the sorted edges.
    for weight, u, v in edges:
        # If adding this edge connects two previously disconnected components,
        # it is part of the MST.
        if dsu.union(u, v):
            mst_weight += weight
            # Optimization: If N-1 edges have been added (or equivalently,
            # the number of components has reduced to 1), the MST is complete.
            if dsu.num_components == 1:
                break # Exit early as MST is fully formed

    # After processing all edges (or early exit):
    # If there is only one component left, the graph is connected,
    # and mst_weight holds the total weight of the MST.
    if dsu.num_components == 1:
        sys.stdout.write(str(mst_weight) + '
')
    else:
        # If more than one component remains, the graph is not connected.
        sys.stdout.write("-1
")

# Call the main solve function
solve()