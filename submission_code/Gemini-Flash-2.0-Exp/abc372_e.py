def solve():
    n, q = map(int, input().split())
    adj = [[] for _ in range(n + 1)]

    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            u, v = query[1], query[2]
            adj[u].append(v)
            adj[v].append(u)
        else:
            v, k = query[1], query[2]
            neighbors = sorted(list(set(adj[v] + [v])), reverse=True)
            if len(neighbors) < k:
                print(-1)
            else:
                print(neighbors[k-1])

solve()