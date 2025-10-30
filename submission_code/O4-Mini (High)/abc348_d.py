def main():
    import sys
    from collections import deque
    input = sys.stdin.readline

    # Read input
    H, W = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]
    # Find start (S) and target (T)
    sx = sy = tx = ty = -1
    for i in range(H):
        row = grid[i]
        for j in range(W):
            c = row[j]
            if c == 'S':
                sx, sy = i, j
            elif c == 'T':
                tx, ty = i, j

    N = int(input())
    # Prepare coordinates list: index 0 is S, 1..N are medicines, N+1 is T
    coords = [(sx, sy)]
    Es = [0] * (N + 2)  # energies, Es[0]=0 for start, Es[1..N] for meds, Es[N+1]=0 for T
    for idx in range(1, N + 1):
        r, c, e = map(int, input().split())
        coords.append((r - 1, c - 1))
        Es[idx] = e
    coords.append((tx, ty))

    # Build a quick index map from grid cell to node index (one per cell)
    # If multiple nodes share a cell (med at S or med at T), the last one in coords wins.
    idxarr = [[-1] * W for _ in range(H)]
    for idx, (x, y) in enumerate(coords):
        idxarr[x][y] = idx

    # We will compute minimal distances on the grid from each source u = 0..N
    INF = H * W + 5
    # dist[u][v] will be the grid-distance from coords[u] to coords[v], or INF if unreachable
    dist = [ [INF] * (N + 2) for _ in range(N + 1) ]

    # Reusable visited array, storing BFS-id stamps
    visited = [[0] * W for _ in range(H)]
    # Directions for grid moves
    dx = (1, -1, 0, 0)
    dy = (0, 0, 1, -1)

    # BFS from each node u = 0..N (we don't BFS from T itself)
    for u in range(N + 1):
        maxd = Es[u]  # we only care about grid-distances up to Es[u]
        mark = u + 1
        x0, y0 = coords[u]
        # Initialize BFS layer
        curr = [(x0, y0)]
        visited[x0][y0] = mark
        d = 0
        # BFS layer by layer up to depth maxd
        while curr:
            next_layer = []
            for (x, y) in curr:
                # Record mapping to any node v at this cell
                v = idxarr[x][y]
                if v >= 0 and dist[u][v] == INF:
                    dist[u][v] = d
                # Expand neighbors only if we haven't reached max depth
                if d < maxd:
                    # Check four directions
                    nx = x + dx[0]; ny = y + dy[0]
                    if 0 <= nx < H and 0 <= ny < W and visited[nx][ny] != mark and grid[nx][ny] != '#':
                        visited[nx][ny] = mark
                        next_layer.append((nx, ny))
                    nx = x + dx[1]; ny = y + dy[1]
                    if 0 <= nx < H and 0 <= ny < W and visited[nx][ny] != mark and grid[nx][ny] != '#':
                        visited[nx][ny] = mark
                        next_layer.append((nx, ny))
                    nx = x + dx[2]; ny = y + dy[2]
                    if 0 <= nx < H and 0 <= ny < W and visited[nx][ny] != mark and grid[nx][ny] != '#':
                        visited[nx][ny] = mark
                        next_layer.append((nx, ny))
                    nx = x + dx[3]; ny = y + dy[3]
                    if 0 <= nx < H and 0 <= ny < W and visited[nx][ny] != mark and grid[nx][ny] != '#':
                        visited[nx][ny] = mark
                        next_layer.append((nx, ny))
            d += 1
            curr = next_layer

    # Build directed graph among the N+2 nodes: edges from u->v if you can
    # use medicine at u (energy Es[u]) then walk to v (grid-dist <= Es[u]).
    # We only have edges from u in [0..N] to v in [1..N+1].
    target_idx = N + 1
    graph = [[] for _ in range(N + 2)]
    for u in range(N + 1):
        eu = Es[u]
        # to medicines 1..N and to target N+1
        # v from 1..N+1
        row = dist[u]
        for v in range(1, N + 2):
            if row[v] <= eu:
                graph[u].append(v)

    # BFS on this directed graph from node 0, looking for node target_idx
    seen = [False] * (N + 2)
    dq = deque([0])
    seen[0] = True
    found = False
    while dq:
        u = dq.popleft()
        if u == target_idx:
            found = True
            break
        for v in graph[u]:
            if not seen[v]:
                seen[v] = True
                dq.append(v)

    print("Yes" if found else "No")

if __name__ == "__main__":
    main()