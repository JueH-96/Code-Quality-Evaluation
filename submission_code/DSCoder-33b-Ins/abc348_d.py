from collections import deque

H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]
N = int(input())
medicines = [list(map(int, input().split())) for _ in range(N)]

for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'T':
            goal = (i, j)

queue = deque([(start[0], start[1], 0)])
visited = [[0]*W for _ in range(H)]
visited[start[0]][start[1]] = 1

while queue:
    x, y, energy = queue.popleft()
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and visited[nx][ny] < energy:
            visited[nx][ny] = energy
            queue.append((nx, ny, energy))
    for r, c, e in medicines:
        if (r-1, c-1) == (x, y) and visited[r-1][c-1] < e:
            visited[r-1][c-1] = e
            queue.append((r-1, c-1, e))

if visited[goal[0]][goal[1]] > 0:
    print('Yes')
else:
    print('No')