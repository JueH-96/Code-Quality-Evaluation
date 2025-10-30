# YOUR CODE HERE
import sys

# Set a higher recursion depth if using a recursive find implementation.
# The iterative version used here avoids deep recursion issues.
# sys.setrecursionlimit(2 * 10**5 + 10) 

def solve():
    """
    Solves the problem of finding the minimum number of edges to delete
    from a graph to make it a forest using the Disjoint Set Union (DSU) data structure.
    """
    # Read number of vertices (N) and edges (M)
    N, M = map(int, sys.stdin.readline().split())

    # Initialize parent array for DSU (using 1-based indexing for vertices)
    # parent[i] stores the parent of node i.
    # Initially, each node is its own parent (representing N disjoint sets).
    parent = list(range(N + 1))
    
    # Initialize size array for DSU optimization (Union by Size)
    # size[i] stores the size of the set rooted at i.
    # Initially, each set contains only one element, so size is 1.
    size = [1] * (N + 1)

    # Iterative find operation with path compression
    # Finds the representative (root) of the set containing node i.
    # Path compression optimizes future find operations by making nodes
    # on the path point directly to the root.
    def find(i):
        # Find the root of the set containing i
        root = i
        while parent[root] != root:
            root = parent[root]
        
        # Path compression: Make all nodes on the path from i to the root
        # point directly to the root. This flattens the tree structure.
        curr = i
        while parent[curr] != root:
            next_node = parent[curr] # Store the next node in the path
            parent[curr] = root      # Point current node to the root
            curr = next_node         # Move to the next node
        return root

    # Union operation with Union by Size optimization
    # Merges the sets containing nodes i and j.
    # It attaches the root of the smaller set to the root of the larger set.
    # Returns True if a merge occurred (i and j were in different sets).
    # Returns False if i and j were already in the same set (adding the edge would create a cycle).
    def union(i, j):
        # Find the roots of the sets containing i and j
        root_i = find(i)
        root_j = find(j)
        
        # If i and j are already in the same set, do nothing and return False
        if root_i != root_j:
            # Union by Size: Attach the smaller tree to the root of the larger tree
            # to keep the tree depth relatively small.
            if size[root_i] < size[root_j]:
                # Make root_j the parent of root_i (root_i becomes child of root_j)
                parent[root_i] = root_j
                # Update the size of the new root's set (root_j)
                size[root_j] += size[root_i]
            else:
                # Make root_i the parent of root_j (root_j becomes child of root_i)
                parent[root_j] = root_i
                # Update the size of the new root's set (root_i)
                size[root_i] += size[root_j]
            return True # A merge was performed successfully
        
        # If roots are the same, i and j are already connected.
        return False # No merge happened.

    # Counter for the number of edges that need to be deleted
    # An edge needs deletion if adding it creates a cycle.
    edges_to_delete = 0
    
    # Process each of the M edges given in the input
    for _ in range(M):
        # Read the endpoints (u, v) of the current edge
        u, v = map(int, sys.stdin.readline().split())
        
        # Attempt to union the sets containing u and v.
        # The union function returns False if u and v are already in the same set.
        # This indicates that adding the edge (u, v) would create a cycle.
        if not union(u, v):
            # If a cycle is detected, increment the counter for edges to delete.
            edges_to_delete += 1

    # Print the final result: the minimum number of edges to remove to make the graph a forest.
    print(edges_to_delete)

# Standard Python execution block to ensure the solve function is called when the script runs.
if __name__ == '__main__':
    solve()