def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    def is_forest(edges_subset, num_nodes):
        if not edges_subset:
            return True
        
        adj = {i: [] for i in range(1, num_nodes + 1)}
        for u, v in edges_subset:
            adj[u].append(v)
            adj[v].append(u)
        
        visited = set()
        
        def dfs(node, parent):
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return True  # Cycle detected
                if dfs(neighbor, node):
                    return True
            return False
        
        for node in range(1, num_nodes + 1):
            if node not in visited:
                if dfs(node, -1):
                    return False
        return True

    
    if m == 0:
        print(0)
        return

    
    count = 0
    
    num_components = 0
    visited = [False] * (n + 1)
    adj = {i: [] for i in range(1, n + 1)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def dfs(node):
        visited[node] = True
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    for i in range(1, n + 1):
        if not visited[i]:
            dfs(i)
            num_components += 1
            
    print(m - (n - num_components))
    
solve()