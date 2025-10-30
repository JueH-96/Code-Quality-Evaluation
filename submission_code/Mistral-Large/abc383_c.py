import sys
from collections import deque

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
D = int(data[2])

grid = [data[i] for i in range(3, 3 + H)]

def is_valid(x, y):
    return 0 <= x < H and 0 <= y < W and grid[x][y] != '#'

def bfs(start_x, start_y):
    queue = deque([(start_x, start_y, 0)])
    visited = set((start_x, start_y))
    humidified = set()

    while queue:
        x, y, dist = queue.popleft()
        if dist > D:
            break
        humidified.add((x, y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

    return humidified

humidified_cells = set()

for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            humidified_cells.update(bfs(i, j))

floor_cells = {(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.'}
humidified_floor_cells = floor_cells & humidified_cells

print(len(humidified_floor_cells))