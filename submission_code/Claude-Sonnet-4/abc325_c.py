def dfs(grid, visited, i, j, H, W):
    if i < 0 or i >= H or j < 0 or j >= W or visited[i][j] or grid[i][j] == '.':
        return
    
    visited[i][j] = True
    
    # Check all 8 directions (horizontal, vertical, diagonal)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            dfs(grid, visited, i + di, j + dj, H, W)

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Initialize visited array
visited = [[False] * W for _ in range(H)]

# Count connected components
components = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#' and not visited[i][j]:
            dfs(grid, visited, i, j, H, W)
            components += 1

print(components)