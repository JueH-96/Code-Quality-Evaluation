import sys
from collections import deque

def solve():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    A, B, C, D = map(int, sys.stdin.readline().split())
    A -= 1
    B -= 1
    C -= 1
    D -= 1

    INF = float('inf')
    dist = [[INF] * W for _ in range(H)]
    dist[A][B] = 0
    q = deque()
    q.append((A, B))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        i, j = q.popleft()
        if i == C and j == D:
            print(dist[i][j])
            return
        # Move to adjacent cells (0 cost)
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.' and dist[ni][nj] > dist[i][j]:
                    dist[ni][nj] = dist[i][j]
                    q.appendleft((ni, nj))
        # Perform kicks (cost 1)
        for di, dj in directions:
            for step in 1, 2:
                ni, nj = i + di * step, j + dj * step
                if 0 <= ni < H and 0 <= nj < W:
                    if dist[ni][nj] > dist[i][j] + 1:
                        dist[ni][nj] = dist[i][j] + 1
                        q.append((ni, nj))
    print(dist[C][D])

solve()