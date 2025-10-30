import sys

def solve():
    # Read N and M from standard input
    N, M = map(int, sys.stdin.readline().split())

    # Initialize DSU data structures
    # parent[i] stores the parent of element i. If parent[i] == i, i is the root of its set.
    # Vertices are 1-indexed, so we create an array of size N+1.
    parent = list(range(N + 1))
    
    # size[i] stores the size of the set if i is the root.
    # This is used for the union-by-size optimization to keep trees flat.
    size = [1] * (N + 1)

    # The find operation with path compression:
    # Recursively finds the root of the set containing element i
    # and flattens the path by making all intermediate nodes point directly to the root.
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i]) # Path compression
        return parent[i]

    # The union operation with union-by-size optimization:
    # Merges the sets containing elements i and j.
    # It attaches the root of the smaller tree to the root of the larger tree.
    def union(i, j):
        root_i = find(i)
        root_j = find(j)

        if root_i != root_j:
            # Union by size: attach the smaller tree under the root of the larger tree
            if size[root_i] < size[root_j]:
                root_i, root_j = root_j, root_i # Swap roots to ensure root_i is the larger one
            
            parent[root_j] = root_i
            size[root_i] += size[root_j]
            return True # Merge successful
        return False # Already in the same set, no merge occurred

    # Initialize the counter for edges that need to be deleted
    edges_to_delete = 0

    # Process each edge in the input graph
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        
        # Check if u and v are already in the same connected component
        if find(u) == find(v):
            # If they are, adding this edge (u, v) would create a cycle.
            # To make the graph a forest, this edge must be one of the deleted ones.
            edges_to_delete += 1
        else:
            # If u and v are in different connected components,
            # adding this edge connects these two components without forming a cycle.
            # This edge becomes part of the spanning forest.
            # Perform the union operation.
            union(u, v)
            
    # Print the total number of edges that need to be deleted
    sys.stdout.write(str(edges_to_delete) + "
")

# Call the solve function to run the program
solve()