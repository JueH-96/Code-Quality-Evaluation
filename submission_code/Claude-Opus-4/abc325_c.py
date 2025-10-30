# YOUR CODE HERE
def dfs(grid, visited, i, j, H, W):
    # Mark current cell as visited
    visited[i][j] = True
    
    # Check all 8 adjacent cells
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for di, dj in directions:
        ni, nj = i + di, j + dj
        # Check if the new position is valid and contains an unvisited sensor
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj] and grid[ni][nj] == '#':
            dfs(grid, visited, ni, nj, H, W)

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Initialize visited array
visited = [[False] * W for _ in range(H)]

# Count connected components
sensor_count = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#' and not visited[i][j]:
            # Found a new sensor group
            dfs(grid, visited, i, j, H, W)
            sensor_count += 1

print(sensor_count)