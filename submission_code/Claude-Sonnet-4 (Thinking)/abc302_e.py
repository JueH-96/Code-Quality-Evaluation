n, q = map(int, input().split())

# Initialize adjacency list
adj = [set() for _ in range(n + 1)]  # 1-indexed

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:  # Add edge
        u, v = query[1], query[2]
        adj[u].add(v)
        adj[v].add(u)
    else:  # Remove all edges connected to vertex v
        v = query[1]
        # Remove v from all its neighbors' adjacency lists
        for neighbor in adj[v]:
            adj[neighbor].remove(v)
        # Clear v's adjacency list
        adj[v].clear()
    
    # Count isolated vertices
    isolated_count = 0
    for i in range(1, n + 1):
        if len(adj[i]) == 0:
            isolated_count += 1
    
    print(isolated_count)