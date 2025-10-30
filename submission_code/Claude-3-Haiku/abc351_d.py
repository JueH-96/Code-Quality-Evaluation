from collections import deque

def max_degree_of_freedom(H, W, grid):
    max_freedom = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                freedom = bfs(H, W, grid, i, j)
                max_freedom = max(max_freedom, freedom)
    return max_freedom

def bfs(H, W, grid, i, j):
    queue = deque([(i, j)])
    visited = set()
    freedom = 1
    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                queue.append((nx, ny))
                freedom += 1
    return freedom

# Read input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Solve the problem
print(max_degree_of_freedom(H, W, grid))