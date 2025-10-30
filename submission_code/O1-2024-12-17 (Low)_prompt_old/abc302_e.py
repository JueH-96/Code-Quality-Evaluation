def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # We'll parse the inputs manually for performance
    # Format:
    # N Q
    # query_1
    # ...
    # query_Q

    # Convert the input tokens to integers
    it = 0
    N = int(input_data[it]); it += 1
    Q = int(input_data[it]); it += 1
    
    # We'll use 0-based indexing internally
    # Keep track of the degree of each vertex
    degree = [0] * N
    
    # Maintain adjacency; for each vertex, store set of neighbors
    adjacency = [set() for _ in range(N)]
    
    # Initially, all vertices are isolated
    isolated_count = N
    
    # Prepare output buffer (to print at the end)
    out = []
    
    for _ in range(Q):
        t = int(input_data[it]); it += 1
        
        if t == 1:
            # query 1 u v
            u = int(input_data[it]) - 1; it += 1
            v = int(input_data[it]) - 1; it += 1
            
            # Increase degree[u], degree[v]
            if degree[u] == 0:
                isolated_count -= 1
            degree[u] += 1
            
            if degree[v] == 0:
                isolated_count -= 1
            degree[v] += 1
            
            # Add edge in adjacency
            adjacency[u].add(v)
            adjacency[v].add(u)
            
        else:
            # query 2 v
            v = int(input_data[it]) - 1
            it += 1
            
            # Remove all edges from v
            neighbors = adjacency[v]
            if neighbors:
                # For each neighbor w
                for w in neighbors:
                    # Decrease degree[w]
                    degree[w] -= 1
                    if degree[w] == 0:
                        isolated_count += 1
                    # Remove v from adjacency[w]
                    adjacency[w].remove(v)
                
                # Clear adjacency[v]
                adjacency[v].clear()
            
            # If v had edges, it now becomes isolated
            if degree[v] > 0:
                degree[v] = 0
                isolated_count += 1
        
        # After each query, print number of isolated vertices
        out.append(str(isolated_count))
    
    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()