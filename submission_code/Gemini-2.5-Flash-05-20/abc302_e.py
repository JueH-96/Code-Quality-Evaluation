import sys

# It's good practice to wrap the main logic in a function.
def solve():
    # Read N and Q from the first line
    N, Q = map(int, sys.stdin.readline().split())

    # Adjacency list: adj[i] is a set of neighbors of vertex i+1 (0-indexed).
    # Using sets provides O(1) average time complexity for add/remove/check existence.
    adj = [set() for _ in range(N)]

    # Initially, all N vertices have no edges, so they are all isolated.
    isolated_count = N

    # Store results to print all at once at the end for efficiency (fewer sys calls).
    results = []

    # Process each query
    for _ in range(Q):
        # Read the current query
        query_parts = list(map(int, sys.stdin.readline().split()))
        q_type = query_parts[0]

        if q_type == 1:
            # Type 1: Connect vertex u and vertex v with an edge
            u, v = query_parts[1], query_parts[2]
            u_idx, v_idx = u - 1, v - 1 # Convert to 0-indexed

            # Before adding the edge, check if u was isolated. If so, it's no longer isolated.
            if len(adj[u_idx]) == 0:
                isolated_count -= 1
            # Check if v was isolated. If so, it's no longer isolated.
            if len(adj[v_idx]) == 0:
                isolated_count -= 1
            
            # Add the edge (u, v) by adding v to u's neighbors and u to v's neighbors.
            adj[u_idx].add(v_idx)
            adj[v_idx].add(u_idx)

        elif q_type == 2:
            # Type 2: Remove all edges connected to vertex v
            v = query_parts[1]
            v_idx = v - 1 # Convert to 0-indexed

            # If vertex v currently has neighbors, it means it was not isolated.
            # After this operation, it will become isolated. So, increment the count.
            # If it had no neighbors (was already isolated), its status doesn't change.
            if len(adj[v_idx]) > 0:
                isolated_count += 1
            
            # We need to iterate over v's neighbors and remove v from their adjacency lists.
            # Create a copy of the neighbors list because we will modify the sets during iteration.
            # Iterating directly over `adj[v_idx]` while modifying `adj[w_idx]` is fine,
            # but modifying `adj[v_idx]` itself (e.g., clearing it) while iterating would be problematic.
            # Making a copy (list(adj[v_idx])) is a robust way to ensure we process all original neighbors.
            neighbors_of_v = list(adj[v_idx]) 

            for w_idx in neighbors_of_v:
                # Remove the edge (v, w) from w's perspective
                adj[w_idx].discard(v_idx)
                
                # After removing v, check if w becomes isolated.
                # If w's adjacency set is now empty, it means w has become isolated.
                if len(adj[w_idx]) == 0:
                    isolated_count += 1
            
            # Finally, clear all edges from v's adjacency list.
            # This ensures v has no outgoing edges and is indeed isolated.
            adj[v_idx].clear()
        
        # After processing each query, append the current isolated count to results.
        results.append(str(isolated_count))
    
    # Print all results, each on a new line, followed by a final newline.
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to run the program
solve()