def main():
    import sys
    input = sys.stdin.readline
    
    # Read N (number of vertices) and Q (number of queries)
    N, Q = map(int, input().split())
    
    # For each vertex 1 to N, we maintain a set of neighbors.
    neighbors = [set() for _ in range(N + 1)]
    
    # Degree for each vertex.
    degree = [0] * (N + 1)
    
    # Initially, all vertices are isolated.
    isolated = N
    
    # Result lines to output after each query.
    res = []
    
    for _ in range(Q):
        query = input().split()
        if not query:
            continue
        
        if query[0] == "1":
            # Query of type "1 u v": add an edge between u and v.
            u = int(query[1])
            v = int(query[2])
            
            # If u (or v) was isolated, then its degree becomes non-zero.
            if degree[u] == 0:
                isolated -= 1
            if degree[v] == 0:
                isolated -= 1
            
            # Add the edge
            neighbors[u].add(v)
            neighbors[v].add(u)
            
            degree[u] += 1
            degree[v] += 1
            
        elif query[0] == "2":
            # Query of type "2 v": remove all edges incident to vertex v.
            v = int(query[1])
            
            # For each neighbor of v, remove the edge v <-> u.
            # We create a list from the set because we'll modify the set while iterating.
            for u in list(neighbors[v]):
                # Remove v from u's neighbor set.
                neighbors[u].remove(v)
                degree[u] -= 1
                # If u becomes isolated, update count.
                if degree[u] == 0:
                    isolated += 1
                    
            # Clear v's neighbor list and update its degree.
            if degree[v] != 0:
                degree[v] = 0
                # v becomes isolated
                isolated += 1
            neighbors[v].clear()
        
        # Append the number of isolated vertices after processing the query.
        res.append(str(isolated))
    
    sys.stdout.write("
".join(res))


if __name__ == '__main__':
    main()