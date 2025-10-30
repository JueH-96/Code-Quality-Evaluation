def humidified_cells(H, W, D, grid):
    from collections import deque

    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Create a visited grid to track humidified cells
    visited = [[False] * W for _ in range(H)]
    queue = deque()

    # Initialize the queue with humidifier positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                queue.append((i, j, 0))  # (row, column, distance)
                visited[i][j] = True

    # BFS to find all humidified cells
    while queue:
        x, y, dist = queue.popleft()
        
        # If we have reached the maximum distance, skip further processing
        if dist < D:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds and if the cell is not a wall and not visited
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny, dist + 1))

    # Count the number of humidified cells
    humidified_count = sum(1 for i in range(H) for j in range(W) if visited[i][j] and grid[i][j] == '.')

    return humidified_count

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W, D = map(int, data[0].split())
grid = data[1:H + 1]

# Calculate and print the result
result = humidified_cells(H, W, D, grid)
print(result)