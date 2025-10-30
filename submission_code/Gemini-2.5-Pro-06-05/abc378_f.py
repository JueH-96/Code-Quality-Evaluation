# YOUR CODE HERE
import sys
from collections import defaultdict

# It is good practice to set a higher recursion limit for problems
# that involve deep recursion, such as graph traversals or DSU on large inputs.
# The default limit in Python is often 1000.
sys.setrecursionlimit(2 * 10**5 + 5)

class DSU:
    """Disjoint Set Union (Union-Find) data structure."""
    def __init__(self, n):
        # parent[i] stores the parent of item i
        self.parent = list(range(n))
        # size[i] stores the size of the set rooted at i
        self.size = [1] * n

    def find(self, i):
        """Find the representative of the set containing item i, with path compression."""
        if self.parent[i] == i:
            return i
        # Path compression: set the parent of i directly to the root
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Merge the sets containing items i and j, using union by size."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size: attach the smaller tree to the root of the larger tree
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True
        return False

def solve():
    """
    Main function to solve the problem.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        N_str = input()
        if not N_str: return
        N = int(N_str)
        
        # Adjacency list to represent the tree
        adj = [[] for _ in range(N + 1)]
        # Array to store the degree of each vertex
        deg = [0] * (N + 1)
        # List of edges to iterate over later
        edges = []

        for _ in range(N - 1):
            u, v = map(int, input().split())
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1

    except (IOError, ValueError):
        return

    # A valid new edge (u, v) requires:
    # 1. deg(u) = 2 and deg(v) = 2 in the original tree.
    # 2. All internal vertices on the path between u and v have degree 3.

    dsu = DSU(N + 1)

    # Step 1: Group all connected degree-3 vertices using DSU.
    for u, v in edges:
        if deg[u] == 3 and deg[v] == 3:
            dsu.union(u, v)

    # Step 2: For each degree-3 component, count its adjacent degree-2 vertices.
    # `endpoints_count` maps the representative of a degree-3 component to its count of degree-2 neighbors.
    endpoints_count = defaultdict(int)

    # Iterate through all vertices. If a vertex has degree 2, it's a potential endpoint.
    for v in range(1, N + 1):
        if deg[v] == 2:
            # Check its neighbors to see if it's connected to a degree-3 component.
            for neighbor in adj[v]:
                if deg[neighbor] == 3:
                    # `v` is an endpoint for the component containing `neighbor`.
                    root = dsu.find(neighbor)
                    endpoints_count[root] += 1
                    # In a tree, a vertex can be adjacent to at most one vertex
                    # from any given connected component of a subgraph. So we can break.
                    break

    # Step 3 & 4: Calculate the total number of valid pairs.
    total_pairs = 0
    for k in endpoints_count.values():
        # For a component with k endpoints, we can choose any 2 to form a valid pair.
        total_pairs += k * (k - 1) // 2
        
    print(total_pairs)


# Call the solve function to run the program
solve()