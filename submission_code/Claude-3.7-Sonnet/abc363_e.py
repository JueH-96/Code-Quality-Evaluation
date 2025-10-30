from collections import deque

def simulate_rising_sea(H, W, Y, elevation):
    # Initialize grid to mark which cells are sunken
    is_sunken = [[False for _ in range(W)] for _ in range(H)]
    
    # For each year, simulate the rising sea
    results = []
    
    for year in range(1, Y + 1):
        # Start BFS from the virtual sea outside the grid
        queue = deque()
        visited = set()
        
        # Add virtual sea cells to the queue
        for i in range(H):
            queue.append((-1, i))   # Left of the grid
            queue.append((W, i))    # Right of the grid
            visited.add((-1, i))
            visited.add((W, i))
        
        for j in range(W):
            queue.append((j, -1))   # Above the grid
            queue.append((j, H))    # Below the grid
            visited.add((j, -1))
            visited.add((j, H))
        
        # BFS to find all cells that will sink in this year
        while queue:
            x, y = queue.popleft()
            
            # Check adjacent cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    
                    # If the cell is already sunken or its elevation is not greater than the sea level, it sinks
                    if is_sunken[ny][nx] or elevation[ny][nx] <= year:
                        is_sunken[ny][nx] = True
                        queue.append((nx, ny))
        
        # Count the remaining land cells
        land_count = H * W - sum(row.count(True) for row in is_sunken)
        results.append(land_count)
    
    return results

# Read input
H, W, Y = map(int, input().split())
elevation = []
for _ in range(H):
    elevation.append(list(map(int, input().split())))

# Solve the problem
results = simulate_rising_sea(H, W, Y, elevation)

# Print results
for result in results:
    print(result)