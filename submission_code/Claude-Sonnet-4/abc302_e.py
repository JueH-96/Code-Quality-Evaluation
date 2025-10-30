# YOUR CODE HERE
N, Q = map(int, input().split())

# Use adjacency list to represent the graph
# adj[i] is a set of vertices connected to vertex i
adj = [set() for _ in range(N + 1)]

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:  # Add edge between u and v
        u, v = query[1], query[2]
        adj[u].add(v)
        adj[v].add(u)
    
    else:  # Remove all edges connected to v
        v = query[1]
        # Remove v from all its neighbors' adjacency lists
        for neighbor in adj[v]:
            adj[neighbor].remove(v)
        # Clear v's adjacency list
        adj[v].clear()
    
    # Count isolated vertices (vertices with no edges)
    isolated_count = 0
    for i in range(1, N + 1):
        if len(adj[i]) == 0:
            isolated_count += 1
    
    print(isolated_count)