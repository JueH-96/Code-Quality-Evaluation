def solve():
    n, m = map(int, input().split())
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj[u - 1].append(v - 1)

    min_cycle_len = float('inf')

    def dfs(u, path, visited):
        nonlocal min_cycle_len
        visited[u] = True
        path.append(u)

        for v in adj[u]:
            if v == 0:
                if 0 in path:
                    cycle_len = len(path)
                    min_cycle_len = min(min_cycle_len, cycle_len)
            elif v in path:
                start_index = path.index(v)
                cycle_len = len(path) - start_index
                min_cycle_len = min(min_cycle_len, cycle_len)
            elif not visited[v]:
                dfs(v, path, visited)

        path.pop()
        visited[u] = False

    visited = [False] * n
    dfs(0, [], visited)

    if min_cycle_len == float('inf'):
        print(-1)
    else:
        print(min_cycle_len)

solve()