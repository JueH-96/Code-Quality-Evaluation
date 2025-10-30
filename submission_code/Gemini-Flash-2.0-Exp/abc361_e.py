def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        edges.append((u - 1, v - 1, w))

    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    total_cost = 0
    leaves = 0
    
    for i in range(n):
        degree = len(adj[i])
        if degree == 1:
            leaves += 1
            
    for u, v, w in edges:
        total_cost += w

    print(2 * total_cost - (sum(sorted([w for u, v, w in edges], reverse=True)[:leaves])))
    
solve()