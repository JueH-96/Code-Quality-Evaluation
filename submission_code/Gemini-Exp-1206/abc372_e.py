def solve():
    n, q = map(int, input().split())
    adj = [set() for _ in range(n + 1)]

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, v = query[1], query[2]
            adj[u].add(v)
            adj[v].add(u)
        else:
            v, k = query[1], query[2]
            connected = list(adj[v])
            connected.append(v)
            connected.sort(reverse=True)
            if len(connected) < k:
                print(-1)
            else:
                print(connected[k - 1])

solve()