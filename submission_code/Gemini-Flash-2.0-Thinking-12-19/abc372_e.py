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

            visited = [False] * (n + 1)
            component = []
            queue = [v]
            visited[v] = True
            while queue:
                u = queue.pop(0)
                component.append(u)
                for neighbor in adj[u]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            component.sort(reverse=True)

            if len(component) < k:
                print(-1)
            else:
                print(component[k-1])

solve()