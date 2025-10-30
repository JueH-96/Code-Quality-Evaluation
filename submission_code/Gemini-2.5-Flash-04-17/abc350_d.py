import sys

# Fast input function
def read_int():
    return int(sys.stdin.readline())

def read_ints():
    return map(int, sys.stdin.readline().split())

# DSU implementation with size and edge count
class DSU:
    def __init__(self, n):
        # Use 1-based indexing for users 1 to N
        self.parent = list(range(n + 1))
        # size[i]: size of the component if i is the root.
        self.size = [1] * (n + 1)
        # edges[i]: number of initial edges within the component if i is the root.
        self.edges = [0] * (n + 1)

    def find(self, i):
        # Finds the root of the set containing element i with path compression
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Unions the sets containing elements i and j, and updates size and edge counts
        root_i = self.find(i)
        root_j = self.find(j)

        # The edge (i, j) is being processed. It contributes 1 to the edge count.
        # This increment happens regardless of whether i and j were already in the same component.
        # If they merge, the edge count of the new component increases by 1.
        # If they were already in the same component, the edge count of that component increases by 1.
        # We perform the union first to find the correct root to add the edge count to.

        if root_i != root_j:
            # Union by size
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i # Merge smaller into larger

            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.edges[root_i] += self.edges[root_j]
            # Add the current edge (i, j) which connected two components.
            self.edges[root_i] += 1
        else:
            # i and j are already in the same component.
            # The edge (i, j) is an internal edge within the component.
            # Increment the edge count for this component.
            self.edges[root_i] += 1

# Read input
N, M = read_ints()

# Initialize DSU
dsu = DSU(N)

# Process initial friendships
for _ in range(M):
    u, v = read_ints()
    # Call union for each edge. The union method handles both merging and internal edge counting.
    dsu.union(u, v)

# Calculate total new friendships
total_new_friendships = 0
# Iterate through all potential roots (1 to N)
for i in range(1, N + 1):
    # If i is the root of a component
    if dsu.parent[i] == i:
        nodes_in_component = dsu.size[i]
        initial_edges_in_component = dsu.edges[i]

        # A connected component, through the repeated operation, becomes a complete graph (a clique).
        # The number of edges in a complete graph (clique) of size K is K * (K - 1) / 2.
        final_edges_in_component = nodes_in_component * (nodes_in_component - 1) // 2

        # The number of new friendships added within this component
        # is the difference between the total edges in the final clique
        # and the initial edges that were already present.
        new_friendships_in_component = final_edges_in_component - initial_edges_in_component

        total_new_friendships += new_friendships_in_component

# Print the result
print(total_new_friendships)