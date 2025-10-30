from collections import deque

def bfs(grid, h, w, start):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque([start])
    visited = set([start])
    freedom = 1

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == '.' and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
                freedom += 1

    return freedom

def max_freedom(grid, h, w):
    max_f = 0

    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                freedom = bfs(grid, h, w, (i, j))
                max_f = max(max_f, freedom)

    return max_f

# Read input
h, w = map(int, input().split())
grid = [input() for _ in range(h)]

# Solve and print output
print(max_freedom(grid, h, w))