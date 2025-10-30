def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))

    if n == 2:
        print(1)
        return

    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    if len(adj[1]) <= 1:
        print(1)
        return

    q = [(1, 0)]
    visited = [False] * (n + 1)
    visited[1] = True
    parent = [0] * (n + 1)
    
    while q:
        u, p = q.pop(0)
        parent[u] = p
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append((v, u))

    
    leaves_count = 0
    
    children = [[] for _ in range(n+1)]
    for i in range(2, n+1):
        children[parent[i]].append(i)
    
    
    
    def dfs(u):
        
        count = 0
        
        
        for v in children[u]:
            count += dfs(v)
        
        if len(children[u]) == 0:
            return 1
        else:
            return count
    
    ans = 0
    for v in adj[1]:
        ans += dfs(v)
    
    print(ans)

solve()