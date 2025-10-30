N, Q = map(int, input().split())
# Keep track of edges for each vertex
edges = [set() for _ in range(N+1)]
# Keep track of isolated vertices
isolated = set(range(1, N+1))

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Add edge between u and v
        u, v = query[1], query[2]
        edges[u].add(v)
        edges[v].add(u)
        # Remove vertices from isolated set if they now have edges
        if u in isolated:
            isolated.remove(u)
        if v in isolated:
            isolated.remove(v)
            
    else:
        # Remove all edges connected to vertex v
        v = query[1]
        # For each vertex connected to v
        for u in edges[v]:
            edges[u].remove(v)
            # If vertex u has no more edges, add it to isolated
            if not edges[u]:
                isolated.add(u)
        # Clear edges of v and add to isolated
        edges[v].clear()
        isolated.add(v)
        
    print(len(isolated))