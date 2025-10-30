import sys

# DSU implementation (Disjoint Set Union)
class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of i. If parent[i] == i, i is the representative.
        # We use 1-based indexing for vertices, so size N+1.
        self.parent = list(range(n + 1))
        # size[i] stores the size of the set rooted at i. Used for union by size.
        self.size = [1] * (n + 1)
        # Initially, each vertex is in its own component.
        self.num_components = n

    def find(self, i):
        # Find the representative of the set containing i.
        # Path compression: make the parent of i the representative directly.
        if self.parent[i] == i:
            return i
        # Store the result of the recursive call to flatten the path
        # This is the standard path compression step
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Union the sets containing i and j.
        root_i = self.find(i)
        root_j = self.find(j)

        # If i and j are already in the same set, do nothing.
        if root_i != root_j:
            # Union by size: attach the smaller tree to the root of the larger tree.
            if self.size[root_i] < self.size[root_j]:
                self.parent[root_i] = root_j
                self.size[root_j] += self.size[root_i]
            else:
                self.parent[root_j] = root_i
                self.size[root_i] += self.size[root_j]

            # Since we merged two distinct sets, the number of components decreases.
            self.num_components -= 1
            return True # Indicate a merge happened
        return False # Indicate they were already connected

# Read N and M from standard input
# Use sys.stdin.readline for faster input, especially for large M
line = sys.stdin.readline().split()
N = int(line[0])
M = int(line[1])

# Initialize DSU structure for N vertices (using 1-based indexing)
dsu = DSU(N)

# Process M edges
for _ in range(M):
    line = sys.stdin.readline().split()
    u = int(line[0])
    v = int(line[1])
    # Perform union operation for each edge (u, v).
    # If u and v were in different components, they are merged.
    # If u and v were already in the same component, adding this edge forms a cycle,
    # but the DSU structure doesn't change and num_components doesn't decrease.
    # The DSU correctly counts the number of components after all edges are processed.
    dsu.union(u, v)

# The number of connected components after processing all edges is stored in dsu.num_components
k = dsu.num_components

# The minimum number of edges to delete to make the graph a forest
# is the total number of edges M minus the maximum number of edges a forest
# with N vertices and k components can have.
# A forest with N vertices and k components has exactly N - k edges (sum of V_i - 1 over k components).
# Number of edges to delete = M - (N - k) = M - N + k
answer = M - N + k

# Print the calculated answer to standard output
print(answer)