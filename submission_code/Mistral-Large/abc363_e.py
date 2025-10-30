import sys
from collections import deque

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
Y = int(data[2])

A = []
index = 3
for i in range(H):
    row = []
    for j in range(W):
        row.append(int(data[index]))
        index += 1
    A.append(row)

def is_valid(x, y):
    return 0 <= x < H and 0 <= y < W

def bfs(sea_level):
    visited = [[False] * W for _ in range(H)]
    queue = deque()

    for i in range(H):
        for j in range(W):
            if A[i][j] <= sea_level and (i == 0 or i == H-1 or j == 0 or j == W-1):
                queue.append((i, j))
                visited[i][j] = True

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and not visited[nx][ny] and A[nx][ny] <= sea_level:
                visited[nx][ny] = True
                queue.append((nx, ny))

    sunk_count = sum(sum(row) for row in visited)
    return H * W - sunk_count

for i in range(1, Y + 1):
    remaining_area = bfs(i)
    print(remaining_area)