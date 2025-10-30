def solve():
    n, q = map(int, input().split())
    adj = [set() for _ in range(n + 1)]
    unconnected_count = n

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, v = query[1], query[2]
            if not adj[u]:
                unconnected_count -= 1
            if not adj[v]:
                unconnected_count -= 1
            adj[u].add(v)
            adj[v].add(u)
        else:
            v = query[1]
            if adj[v]:
                unconnected_count += 1
            for u in adj[v]:
                adj[u].remove(v)
                if not adj[u]:
                    unconnected_count += 1
            adj[v].clear()
        
        print(unconnected_count)

solve()