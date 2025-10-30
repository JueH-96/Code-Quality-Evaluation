from collections import defaultdict, deque

def solve():
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
        return
    
    # Find parent and depth for each node using BFS from node 1
    parent = [-1] * (N + 1)
    depth = [0] * (N + 1)
    visited = [False] * (N + 1)
    
    queue = deque([1])
    visited[1] = True
    
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)
    
    # Find LCA of all specified vertices
    def find_lca(u, v):
        # Make u and v at same depth
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
    lca = specified[0]
    for i in range(1, K):
        lca = find_lca(lca, specified[i])
    
    # Mark all vertices on paths from specified vertices to LCA
    marked = set()
    
    for vertex in specified:
        current = vertex
        while current != -1:
            marked.add(current)
            if current == lca:
                break
            current = parent[current]
    
    print(len(marked))

solve()