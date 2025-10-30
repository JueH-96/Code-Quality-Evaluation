# YOUR CODE HERE
import sys
import itertools

# Function to perform the core logic
def solve():
    # Read input values N, M, K
    N, M, K = map(int, sys.stdin.readline().split())
    
    # Store edges. Each edge is represented as a dictionary containing its endpoints u, v and weight w.
    # Using a list of dictionaries for clarity, though tuples (u, v, w) would also work.
    all_edges = []
    for i in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        # Vertices are 1-based in the input, store them as is.
        all_edges.append({'u': u, 'v': v, 'w': w})

    # Base case: If the graph has only one vertex (N=1), a spanning tree has 0 edges and its cost is 0.
    if N == 1:
        print(0)
        return

    # Base case: If K=1, the cost of any spanning tree is (sum of weights) % 1 = 0.
    # Since the graph is connected, at least one spanning tree exists. The minimum cost must be 0.
    if K == 1:
        print(0)
        return

    # Initialize the minimum cost found so far. We use K as the initial value.
    # Since any valid cost must be in the range [0, K-1], any actual cost found will be smaller than K.
    min_cost_mod_K = K 

    # Iterate through all possible combinations of N-1 edges from the M available edges.
    # `itertools.combinations(range(M), N - 1)` generates tuples of indices of the edges.
    for edge_indices_combo in itertools.combinations(range(M), N - 1):
        
        # Get the actual edge objects (dictionaries) for the current combination of indices.
        current_combo_edges = [all_edges[i] for i in edge_indices_combo]
        
        # Check if this combination of N-1 edges forms a spanning tree.
        # We use the Disjoint Set Union (DSU) data structure for this check.
        
        # Initialize the parent array for DSU. `parent[i]` stores the parent of vertex `i`.
        # We use indices 1 to N for vertices, so the array size is N+1. Index 0 is unused.
        # Initially, each vertex is its own parent (representing N disjoint sets).
        parent = list(range(N + 1)) 
        
        # DSU `find_set` operation with path compression optimization.
        # It finds the representative (root) of the set containing vertex `v_`.
        # Path compression makes subsequent finds faster by updating parent pointers along the path to point directly to the root.
        def find_set(v_):
            # If v_ is its own parent, it's the root.
            if v_ == parent[v_]:
                return v_
            # Recursively find the root and update parent[v_] to point directly to it (path compression).
            parent[v_] = find_set(parent[v_]) 
            return parent[v_]

        # DSU `unite_sets` operation. It merges the sets containing vertices `a_` and `b_`.
        # Returns True if a merge occurred (i.e., `a_` and `b_` were in different sets), False otherwise.
        def unite_sets(a_, b_):
            # Find the roots of the sets containing a_ and b_.
            a_root = find_set(a_)
            b_root = find_set(b_)
            # If the roots are different, the vertices are in different sets. Merge them.
            if a_root != b_root:
                # Perform the union by making one root the parent of the other.
                # A simple strategy (e.g., always make b_root's parent a_root) is sufficient for small N.
                parent[b_root] = a_root 
                return True # Union was successful
            # If roots are the same, they are already in the same set. Do nothing.
            return False # No union performed

        # Apply the `unite_sets` operation for all edges in the current combination.
        # This simulates adding these edges to an initially empty graph and tracking connected components using DSU.
        for edge in current_combo_edges:
             unite_sets(edge['u'], edge['v'])

        # After processing all N-1 edges, check if the resulting graph is connected.
        # A graph with N vertices and N-1 edges is a tree (and thus a spanning tree) if and only if it is connected.
        # To check connectivity, verify that all vertices belong to the same component (set in DSU).
        
        # We need to handle the case N=0, although problem constraints state N >= 2.
        # If N > 0, check connectivity starting from vertex 1.
        if N > 0:
             # Find the representative root for vertex 1.
            root_node = find_set(1) 
            is_connected = True
            # Iterate through all other vertices (from 2 to N).
            for i in range(2, N + 1):
                # If any vertex has a different root than vertex 1, the graph is not connected.
                if find_set(i) != root_node:
                     is_connected = False
                     break
        else: 
             # This case should not occur based on constraints N >= 2.
             is_connected = True # An empty graph could be considered connected vacuously.

        # If the graph formed by the selected N-1 edges is connected, it is a spanning tree.
        if is_connected:
             # Calculate the total weight of the edges in this spanning tree.
             current_total_weight = 0
             for edge in current_combo_edges:
                 current_total_weight += edge['w']

             # Calculate the cost of this spanning tree: total weight modulo K.
             # Python handles large integers automatically, so overflow is not an issue.
             current_cost_mod_K = current_total_weight % K
             
             # Update the overall minimum cost found so far across all spanning trees checked.
             min_cost_mod_K = min(min_cost_mod_K, current_cost_mod_K)

    # After iterating through all combinations of N-1 edges, `min_cost_mod_K` holds the minimum cost.
    # Since the input graph is guaranteed to be connected, at least one spanning tree exists,
    # so `min_cost_mod_K` will have been updated from its initial value K (unless K=1, handled earlier).
    print(min_cost_mod_K)

# Execute the main function to solve the problem based on standard input/output.
solve()