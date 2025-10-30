# YOUR CODE HERE
N, Q = map(int, input().split())

# Initialize adjacency list - each vertex has a set of connected vertices
adj = [set() for _ in range(N + 1)]

# Count of isolated vertices (vertices with no edges)
isolated_count = N

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1: Add edge between u and v
        u, v = query[1], query[2]
        
        # If u was isolated before, it's no longer isolated
        if len(adj[u]) == 0:
            isolated_count -= 1
        
        # If v was isolated before, it's no longer isolated
        if len(adj[v]) == 0:
            isolated_count -= 1
        
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
    
    else:
        # Type 2: Remove all edges connected to vertex v
        v = query[1]
        
        # For each neighbor of v, remove v from their adjacency list
        for neighbor in adj[v]:
            adj[neighbor].remove(v)
            # If neighbor becomes isolated, increment count
            if len(adj[neighbor]) == 0:
                isolated_count += 1
        
        # If v had connections, it becomes isolated now
        if len(adj[v]) > 0:
            isolated_count += 1
        
        # Clear v's adjacency list
        adj[v].clear()
    
    print(isolated_count)