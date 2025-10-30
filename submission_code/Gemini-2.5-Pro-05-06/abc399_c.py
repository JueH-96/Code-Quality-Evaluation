import sys

# Set a higher recursion limit for DSU's recursive find operation.
# While union-by-size/rank keeps average depth O(log N), worst-case single find on an
# uncompressed path can be O(N). For N=2e5, this needs a higher limit.
# A limit around N + buffer is typical. Using 2.5e5 should be safe for N up to 2e5.
sys.setrecursionlimit(2 * 10**5 + 5000) 

# Function to read space-separated integers from a line for faster I/O
def read_ints():
    return map(int, sys.stdin.readline().split())

def solve():
    N, M = read_ints()

    parent = list(range(N + 1)) # parent[i] is parent of element i (1-indexed)
    # set_size[i] stores the size of the set rooted at i.
    # This is only meaningful if parent[i] == i (i.e., i is a root).
    set_size = [1] * (N + 1) 

    def find(i):
        # Find the root of the set containing element i
        # with path compression heuristic.
        if parent[i] == i:
            return i
        # Path compression: make a_val node point directly to the root.
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        # Union sets containing elements i and j
        # using union by size heuristic.
        # Returns True if sets were merged, False if they were already the same set.
        root_i = find(i)
        root_j = find(j)

        if root_i != root_j:
            # Attach smaller tree under root of larger tree's root.
            # This helps keep the trees relatively flat.
            if set_size[root_i] < set_size[root_j]:
                # Swap roots so root_i is always the root of the larger (or equal sized) tree
                root_i, root_j = root_j, root_i 
            
            parent[root_j] = root_i
            set_size[root_i] += set_size[root_j]
            return True # Successfully merged
        return False # Already in the same set; adding this edge would form a cycle

    edges_to_remove_count = 0
    for _ in range(M):
        u, v = read_ints()
        if not union(u, v):
            # If u and v are already in the same set (i.e., find(u) == find(v)),
            # adding the edge (u,v) would form a cycle with existing paths.
            # Such an edge must be removed to make the graph a forest.
            edges_to_remove_count += 1
            
    sys.stdout.write(str(edges_to_remove_count) + "
")

if __name__ == '__main__':
    solve()