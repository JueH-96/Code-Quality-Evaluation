# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

# Keep track of visited sensors
visited = [[False] * W for _ in range(H)]

# Directions for 8-connected neighbors (horizontal, vertical, diagonal)
directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def dfs(row, col):
    # Mark current cell as visited
    visited[row][col] = True
    
    # Check all 8 neighbors
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        
        # Check if neighbor is within bounds
        if 0 <= new_row < H and 0 <= new_col < W:
            # If neighbor has a sensor and hasn't been visited
            if grid[new_row][new_col] == '#' and not visited[new_row][new_col]:
                dfs(new_row, new_col)

# Count connected components
sensor_groups = 0

for i in range(H):
    for j in range(W):
        # If this cell has a sensor and hasn't been visited
        if grid[i][j] == '#' and not visited[i][j]:
            # Start a new DFS from this sensor
            dfs(i, j)
            sensor_groups += 1

print(sensor_groups)