from collections import deque

# Read input
H, W, D = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find all humidifier positions
humidifiers = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            humidifiers.append((i, j))

# If no humidifiers, answer is 0
if not humidifiers:
    print(0)
else:
    # Multi-source BFS
    queue = deque()
    visited = set()
    
    # Start from all humidifier positions
    for i, j in humidifiers:
        queue.append((i, j, 0))  # (row, col, distance)
        visited.add((i, j))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        row, col, dist = queue.popleft()
        
        # If we've reached maximum distance, don't explore further
        if dist >= D:
            continue
            
        # Explore neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < H and 0 <= new_col < W:
                # Check if not visited and not a wall
                if (new_row, new_col) not in visited and grid[new_row][new_col] != '#':
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, dist + 1))
    
    # Count humidified floor cells (including humidifier cells)
    humidified_count = 0
    for row, col in visited:
        if grid[row][col] in ['H', '.']:
            humidified_count += 1
    
    print(humidified_count)