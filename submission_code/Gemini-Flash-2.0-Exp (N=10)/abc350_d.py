def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    ans = 0
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            if j in adj[i]:
                for k in range(1, n + 1):
                    if k == i or k == j:
                        continue
                    if k in adj[j] and k not in adj[i]:
                        ans += 1
    
    print(ans // 2)

solve()