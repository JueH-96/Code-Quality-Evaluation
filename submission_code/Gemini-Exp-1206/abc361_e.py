def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        a, b, c = map(int, input().split())
        edges.append((a, b, c))

    adj = [[] for _ in range(n + 1)]
    for a, b, c in edges:
        adj[a].append((b, c))
        adj[b].append((a, c))

    total_cost = 0
    max_leaf_edge = 0

    def dfs(u, p):
        nonlocal max_leaf_edge
        is_leaf = True
        max_edge = 0
        for v, c in adj[u]:
            if v != p:
                is_leaf = False
                dfs(v, u)
                max_edge = max(max_edge, c)
        
        if is_leaf:
            max_leaf_edge = max(max_leaf_edge, 0)
        else:
            max_leaf_edge = max(max_leaf_edge, max_edge)

    for a, b, c in edges:
        total_cost += c

    dfs(1, 0)
    
    print(total_cost * 2 - max_leaf_edge)

solve()