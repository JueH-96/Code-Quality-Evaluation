import sys
from collections import deque

H, W = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(H)]

if grid[0][0] != 's':
    print("No")
    sys.exit()

visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]
queue = deque()
queue.append((0, 0, 0))
visited[0][0][0] = True

dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
snuke = 'snuke'

while queue:
    i, j, p = queue.popleft()
    if i == H - 1 and j == W - 1:
        print("Yes")
        sys.exit()
    next_p = (p + 1) % 5
    for di, dj in dirs:
        ni = i + di
        nj = j + dj
        if 0 <= ni < H and 0 <= nj < W:
            if grid[ni][nj] == snuke[next_p] and not visited[ni][nj][next_p]:
                visited[ni][nj][next_p] = True
                queue.append((ni, nj, next_p))

print("No")