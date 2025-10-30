# YOUR CODE HERE
from collections import deque

def bfs(grid, start, D):
    H, W = len(grid), len(grid[0])
    queue = deque([(start[0], start[1], 0)])
    visited = set([(start[0], start[1])])
    humidified = set([(start[0], start[1])])
    
    while queue:
        x, y, dist = queue.popleft()
        if dist == D:
            continue
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited and grid[nx][ny] != '#':
                visited.add((nx, ny))
                humidified.add((nx, ny))
                queue.append((nx, ny, dist + 1))
    
    return humidified

# Read input
H, W, D = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Find humidifiers
humidifiers = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == 'H']

# Perform BFS from each humidifier
humidified = set()
for hum in humidifiers:
    humidified |= bfs(grid, hum, D)

# Count humidified floor cells
count = sum(1 for i, j in humidified if grid[i][j] in '.H')

# Print result
print(count)