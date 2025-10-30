# YOUR CODE HERE
import sys

# It's good practice in competitive programming to increase the recursion limit
# for algorithms like DSU on large inputs, as the 'find' operation is recursive.
sys.setrecursionlimit(2 * 10**5 + 5)

def main():
    """
    Reads a graph from stdin and computes the minimum number of edges to delete
    to make it a forest.
    """
    try:
        # Fast I/O for large inputs
        input = sys.stdin.readline
        line = input()
        if not line.strip():
            # Handle empty input case (not expected by problem constraints but robust)
            print(0)
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        # Gracefully handle empty input or formatting errors.
        # For N=0, M=0, the answer is 0.
        print(0)
        return

    # A graph is a forest if it's acyclic. An edge (u, v) creates a cycle if
    # vertices u and v are already in the same connected component. We can
    # count such edges using a Disjoint Set Union (DSU) data structure.

    # --- DSU (Union-Find) Implementation ---
    # `parent[i]` stores the parent of node i. A node is a root if parent[i] == i.
    # Vertices are 1-indexed, so we use an array of size N+1.
    parent = list(range(N + 1))

    def find(i):
        """Finds the root of the set containing node i, with path compression."""
        if parent[i] == i:
            return i
        # Path compression: Attach this node directly to the root.
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        """Merges the sets containing nodes i and j."""
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
    # --- End of DSU Implementation ---

    # This counter will store the number of edges that form cycles.
    edges_to_delete = 0

    # Process each of the M edges from the input.
    for _ in range(M):
        u, v = map(int, input().split())

        # If u and v are already in the same component (have the same root),
        # this edge is redundant and creates a cycle. It must be "deleted".
        if find(u) == find(v):
            edges_to_delete += 1
        else:
            # Otherwise, this edge connects two different components. It's a
            # valid edge in the forest we are building. We merge the components
            # by uniting their sets.
            union(u, v)

    # The total number of edges to delete is the number of cycle-forming edges.
    print(edges_to_delete)

if __name__ == "__main__":
    main()