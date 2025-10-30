def find_humidified_cells(H, W, D, grid):
    # Find all humidifier positions
    humidifiers = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                humidifiers.append((i, j))
    
    # Set to store all humidified cells
    humidified = set()
    
    # Directions for movement: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # For each humidifier, do BFS to find all reachable cells within D moves
    for start_i, start_j in humidifiers:
        # Queue for BFS: (row, col, distance)
        queue = [(start_i, start_j, 0)]
        # Set to keep track of visited cells for this humidifier
        visited = {(start_i, start_j)}
        
        while queue:
            i, j, dist = queue.pop(0)
            # Add current cell to humidified set
            humidified.add((i, j))
            
            # If we've reached maximum distance, don't explore further
            if dist == D:
                continue
                
            # Try all four directions
            for di, dj in directions:
                new_i, new_j = i + di, j + dj
                
                # Check if new position is valid
                if (0 <= new_i < H and 0 <= new_j < W and 
                    grid[new_i][new_j] != '#' and 
                    (new_i, new_j) not in visited):
                    queue.append((new_i, new_j, dist + 1))
                    visited.add((new_i, new_j))
    
    # Count humidified floor cells (including cells with humidifiers)
    count = 0
    for i, j in humidified:
        if grid[i][j] in '.H':
            count += 1
            
    return count

# Read input
H, W, D = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input().strip()))

# Calculate and print result
result = find_humidified_cells(H, W, D, grid)
print(result)