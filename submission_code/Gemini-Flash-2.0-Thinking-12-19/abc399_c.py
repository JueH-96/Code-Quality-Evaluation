import sys

# DSU structure
class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i
        # Initially, each element is its own parent
        # Using 1-based indexing for vertices
        self.parent = list(range(n + 1))
        # Stores the size/rank of the set rooted at i
        # Used for union by size optimization
        self.size = [1] * (n + 1)
        # Number of disjoint sets/components
        self.num_components = n

    def find(self, i):
        # Find the root of the set containing element i
        # With path compression
        if self.parent[i] == i:
            return i
        # Path compression: set parent[i] to the root
        # This is done recursively, effectively flattening the tree
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Union the sets containing elements i and j
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by size: attach the smaller tree to the root of the larger tree
            if self.size[root_i] < self.size[root_j]:
                self.parent[root_i] = root_j
                self.size[root_j] += self.size[root_i]
            else:
                self.parent[root_j] = root_i
                self.size[root_i] += self.size[root_j]

            # Decrease the number of components if two distinct components were merged
            self.num_components -= 1
            # Return value not strictly needed for this problem's logic, but can be used
            # to count edges that merge components (which equals N-k at the end)
            # or edges that don't merge (which equals M - (N-k) = M - N + k)
            # return True # Components were merged
        # else:
            # return False # They were already in the same component


# Read input
# sys.stdin.readline is faster than input() for large inputs
line = sys.stdin.readline().split()
n = int(line[0])
m = int(line[1])

# Initialize DSU structure with N elements (vertices 1 to N)
dsu = DSU(n)

# Process M edges
for _ in range(m):
    line = sys.stdin.readline().split()
    u = int(line[0])
    v = int(line[1])
    # The union operation automatically merges components
    # The DSU structure tracks the number of components internally.
    # We don't need to explicitly check if union merged components for the final formula M - N + k.
    dsu.union(u, v)

# After processing all edges, dsu.num_components holds the final number of connected components (k)
k = dsu.num_components

# The minimum number of edges to delete to make the graph a forest
# is the number of edges that are "excess" beyond a spanning forest.
# A spanning forest of a graph with N vertices and k components has exactly N - k edges.
# The given graph has M edges.
# The number of edges we must remove is the difference: M - (N - k).
result = m - n + k

# Print the answer to standard output
print(result)