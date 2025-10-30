import sys
from collections import deque

def solve():
    H, W, Y = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]

    for _ in range(Y):
        # Increase sea level
        sea_level = max(map(max, A)) + 1

        # Check and sink sections
        visited = [[False]*W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                if A[i][j] >= sea_level or visited[i][j]:
                    continue
                visited[i][j] = True
                queue = deque([(i, j)])
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if nx < 0 or nx >= H or ny < 0 or ny >= W or visited[nx][ny] or A[nx][ny] >= sea_level:
                            continue
                        visited[nx][ny] = True
                        queue.append((nx, ny))

        # Count remaining sections
        remaining = sum(row.count(False) for row in visited)
        print(remaining)

solve()