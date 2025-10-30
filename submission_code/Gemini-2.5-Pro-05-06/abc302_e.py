import sys

def main():
    # Use sys.stdin.readline for faster input
    _input = sys.stdin.readline
    
    N, Q = map(int, _input().split())

    # Adjacency lists for each vertex (0-indexed)
    adj = [set() for _ in range(N)]
    
    # Degree of each vertex (0-indexed)
    degree = [0] * N
    
    # Count of isolated vertices (degree == 0)
    # Initially all N vertices are isolated
    isolated_count = N
    
    # To store results of each query (as strings for efficient joining)
    results = []

    for _ in range(Q):
        query_parts = list(map(int, _input().split()))
        query_type = query_parts[0]

        if query_type == 1:
            # Add edge between u and v
            u, v = query_parts[1], query_parts[2]
            u -= 1  # Adjust to 0-indexed
            v -= 1  # Adjust to 0-indexed

            # If u was isolated (degree == 0), it's no longer isolated
            if degree[u] == 0:
                isolated_count -= 1
            degree[u] += 1
            adj[u].add(v)

            # If v was isolated (degree == 0), it's no longer isolated
            if degree[v] == 0:
                isolated_count -= 1
            degree[v] += 1
            adj[v].add(u)
        
        elif query_type == 2:
            # Remove all edges connected to vertex v_val
            v_val = query_parts[1]
            v_idx = v_val - 1 # Adjust to 0-indexed
            
            # For each neighbor w_neighbor of v_idx:
            #   - Remove v_idx from w_neighbor's adjacency list.
            #   - Decrement w_neighbor's degree.
            #   - If w_neighbor becomes isolated, increment isolated_count.
            # Note: degree[v_idx] is not changed yet. adj[v_idx] is iterated but not modified here.
            for w_neighbor in adj[v_idx]:
                adj[w_neighbor].remove(v_idx) 
                degree[w_neighbor] -= 1
                if degree[w_neighbor] == 0:
                    isolated_count += 1
            
            # Now, handle v_idx itself.
            # If v_idx had any edges (its degree was > 0 before its edges were removed),
            # it means v_idx was not isolated. Now it becomes isolated.
            # The value of degree[v_idx] here is its degree *before* being set to 0.
            if degree[v_idx] > 0:
                isolated_count += 1 # v_idx itself transitions from non-isolated to isolated
            
            # Set v_idx's degree to 0 and clear its adjacency list.
            degree[v_idx] = 0
            adj[v_idx] = set() 
            
        results.append(str(isolated_count))

    # Print all results, each on a new line
    sys.stdout.write("
".join(results) + "
")

if __name__ == '__main__':
    main()