from collections import deque

def is_humidified(grid, i, j, D):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    queue = deque([(i, j)])
    visited[i][j] = True
    steps = 0
    while queue:
        for _ in range(len(queue)):
            x, y = queue.popleft()
            if grid[x][y] == 'H':
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        steps += 1
        if steps > D:
            return False
    return False

def count_humidified_cells(grid, H, W, D):
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and is_humidified(grid, i, j, D):
                count += 1
    return count

# Read input
H, W, D = map(int, input().split())
grid = [input() for _ in range(H)]

# Solve the problem
result = count_humidified_cells(grid, H, W, D)

# Write output
print(result)