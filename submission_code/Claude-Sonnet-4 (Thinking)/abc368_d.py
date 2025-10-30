from collections import defaultdict, deque

N, K = map(int, input().split())

# Build adjacency list
adj = defaultdict(list)
for _ in range(N - 1):
    a, b = map(int, input().split())
    adj[a].append(b)
    adj[b].append(a)

specified = list(map(int, input().split()))

if K == 1:
    print(1)
else:
    # Root the tree at vertex 1
    root = 1
    parent = [0] * (N + 1)
    depth = [0] * (N + 1)
    
    # BFS to find parent and depth
    queue = deque([root])
    visited = set([root])
    parent[root] = -1  # Root has no parent
    
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)
    
    # Find LCA of two vertices
    def lca(u, v):
        # Make u and v at the same depth
        while depth[u] > depth[v]:
            u = parent[u]
        while depth[v] > depth[u]:
            v = parent[v]
        
        # Move both up until they meet
        while u != v:
            u = parent[u]
            v = parent[v]
        
        return u
    
    # Find LCA of all specified vertices
    current_lca = specified[0]
    for i in range(1, K):
        current_lca = lca(current_lca, specified[i])
    
    # Include all vertices on the path from each specified vertex to the LCA
    steiner_vertices = set()
    
    for vertex in specified:
        current = vertex
        while True:
            steiner_vertices.add(current)
            if current == current_lca:
                break
            current = parent[current]
    
    print(len(steiner_vertices))