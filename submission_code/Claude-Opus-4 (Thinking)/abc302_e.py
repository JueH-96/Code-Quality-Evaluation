n, q = map(int, input().split())

# Adjacency list representation
adj = [set() for _ in range(n + 1)]
isolated_count = n  # Initially all vertices are isolated

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1: Connect u and v
        u, v = query[1], query[2]
        
        # If u was isolated, it's no longer isolated
        if len(adj[u]) == 0:
            isolated_count -= 1
        # If v was isolated, it's no longer isolated
        if len(adj[v]) == 0:
            isolated_count -= 1
        
        adj[u].add(v)
        adj[v].add(u)
    else:
        # Type 2: Remove all edges from vertex v
        v = query[1]
        
        # If v wasn't isolated, it will become isolated
        if len(adj[v]) > 0:
            isolated_count += 1
        
        # Remove edges from all neighbors
        for neighbor in adj[v]:
            adj[neighbor].remove(v)
            # If neighbor becomes isolated, increment count
            if len(adj[neighbor]) == 0:
                isolated_count += 1
        
        # Clear v's adjacency list
        adj[v].clear()
    
    print(isolated_count)