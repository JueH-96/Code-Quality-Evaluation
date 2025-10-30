def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))

    def dfs(u, target_color, visited, current_color):
        visited[u] = True
        for v in [u - 1, u + 1]:
            if 0 <= v < n and not visited[v] and colors[v] == current_color:
                dfs(v, target_color, visited, current_color)
        colors[u] = target_color

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, c = query[1] -1, query[2]
            visited = [False] * n
            dfs(x, c, visited, colors[x])
        else:
            c = query[1]
            print(colors.count(c))

solve()