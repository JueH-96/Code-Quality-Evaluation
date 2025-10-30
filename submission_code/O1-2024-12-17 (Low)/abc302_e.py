def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster pointer-based input reading
    # Format of input:
    # N Q
    # query_1
    # query_2
    # ...
    # query_Q
    # where each query is either:
    #   1 u v  (connect u and v)
    #   2 v    (remove all edges connected to v)

    # Parse first line
    N = int(input_data[0])
    Q = int(input_data[1])

    # Adjacency list (using sets for flexible edge removal)
    adjacency = [set() for _ in range(N + 1)]
    # Degree array (track how many edges connect to each vertex)
    degree = [0] * (N + 1)

    # Initially all vertices are isolated
    isolated_count = N  

    # Prepare to process queries
    ptr = 2  # pointer in input_data (we've consumed first two integers N, Q)

    outputs = []
    
    for _ in range(Q):
        t = int(input_data[ptr]); ptr += 1
        if t == 1:
            # query: 1 u v
            u = int(input_data[ptr]); ptr += 1
            v = int(input_data[ptr]); ptr += 1
            # If either is isolated before, we reduce the isolated count
            if degree[u] == 0:
                isolated_count -= 1
            if degree[v] == 0:
                isolated_count -= 1
            # Connect u and v
            adjacency[u].add(v)
            adjacency[v].add(u)
            degree[u] += 1
            degree[v] += 1
            
        else:
            # query: 2 v
            v = int(input_data[ptr]); ptr += 1
            # If v was not isolated, it will become isolated
            if degree[v] > 0:
                isolated_count += 1
            # Remove edges from v to each neighbor
            neighbors = adjacency[v]
            for w in list(neighbors):
                adjacency[w].remove(v)
                degree[w] -= 1
                # If a neighbor lost its last connection, it becomes isolated
                if degree[w] == 0:
                    isolated_count += 1
            # Clear v's adjacency
            adjacency[v].clear()
            degree[v] = 0

        # After processing this query, append the current count of isolated vertices
        outputs.append(str(isolated_count))

    # Print all answers
    print("
".join(outputs))

# Call main() at the end
if __name__ == "__main__":
    main()