def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, c = query[1], query[2]
            x -= 1
            original_color = colors[x]
            if original_color == c:
                continue

            reachable_cells = []
            visited = [False] * n
            q_bfs = [x]
            visited[x] = True

            while q_bfs:
                u = q_bfs.pop(0)
                reachable_cells.append(u)
                for v in [u - 1, u + 1]:
                    if 0 <= v < n and not visited[v] and colors[v] == original_color:
                        visited[v] = True
                        q_bfs.append(v)

            for cell_index in reachable_cells:
                colors[cell_index] = c
        elif query[0] == 2:
            c = query[1]
            count = 0
            for color in colors:
                if color == c:
                    count += 1
            print(count)

solve()