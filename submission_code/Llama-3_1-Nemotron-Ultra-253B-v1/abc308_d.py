import sys
from collections import deque

h, w = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(h)]

target = ['s', 'n', 'u', 'k', 'e']

if grid[0][0] != target[0]:
    print("No")
    exit()

visited = [[[False] * 5 for _ in range(w)] for __ in range(h)]

q = deque()
q.append((0, 0, 0))
visited[0][0][0] = True

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

while q:
    i, j, m = q.popleft()
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            next_m = (m + 1) % 5
            if grid[ni][nj] == target[next_m] and not visited[ni][nj][next_m]:
                visited[ni][nj][next_m] = True
                q.append((ni, nj, next_m))

end_i, end_j = h - 1, w - 1
for m in range(5):
    if visited[end_i][end_j][m]:
        print("Yes")
        exit()

print("No")