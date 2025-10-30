# YOUR CODE HERE
import sys

# Set recursion depth for DSU path compression
# For N up to 2e5, path compression can lead to deep recursion.
# Union by size/rank keeps tree height logarithmic, so path compression depth
# is also limited. A limit larger than N is generally safe.
sys.setrecursionlimit(300000)

# Disjoint Set Union (DSU) structure
class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i
        # Using 1-based indexing for vertices 1 to n
        self.parent = list(range(n + 1))
        # size[i] stores the size of the set rooted at i
        self.size = [1] * (n + 1)

    # Find the representative (root) of the set containing element i
    # with path compression
    def find(self, i):
        if self.parent[i] == i:
            return i
        # Path compression: set parent of i to the root
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    # Union the sets containing elements i and j
    # using union by size
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size: attach smaller tree to larger tree
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True  # Components were merged
        return False # Already in the same component

# Read input
# Use sys.stdin.readline for faster input
input = sys.stdin.readline
N, M = map(int, input().split())

# Initialize DSU for the initial graph G
dsu = DSU(N)

# Process initial edges of G
for _ in range(M):
    u, v = map(int, input().split())
    # The DSU handles multi-edges and self-loops gracefully.
    # If u == v, union(u,u) correctly does nothing.
    dsu.union(u, v)

# Read K constraints
K = int(input())

# Store forbidden pairs of component representatives
# If an edge (p, q) connects components root_p and root_q,
# and {root_p, root_q} is in this set, then the graph becomes not good.
# Use a set of sorted tuples for efficient lookup.
forbidden_pairs = set()

for _ in range(K):
    x, y = map(int, input().split())
    # Find component representatives for x and y in graph G
    root_x = dsu.find(x)
    root_y = dsu.find(y)
    
    # The problem guarantees x and y are disconnected in G, so root_x != root_y
    # Store the pair of representatives (sorted)
    # Sorting ensures {a, b} and {b, a} map to the same key (min(a,b), max(a,b))
    forbidden_pairs.add(tuple(sorted((root_x, root_y))))

# Read Q queries
Q = int(input())

# Process queries
for _ in range(Q):
    p, q = map(int, input().split())
    # Find component representatives for p and q in graph G
    root_p = dsu.find(p)
    root_q = dsu.find(q)

    # If p and q are already in the same component in G, adding edge (p,q)
    # does not change connectivity between different components. G_i is good.
    if root_p == root_q:
        print("Yes")
    else:
        # If p and q are in different components, adding edge (p,q) merges
        # their components. The graph G_i is NOT good if this merger connects
        # any previously disconnected constraint pair (x_j, y_j).
        # This happens if the components being merged are exactly the
        # components containing x_j and y_j (in G).
        # Check if the pair of components {root_p, root_q} is in the
        # forbidden_pairs set.
        query_pair = tuple(sorted((root_p, root_q)))
        if query_pair in forbidden_pairs:
            print("No")
        else:
            print("Yes")