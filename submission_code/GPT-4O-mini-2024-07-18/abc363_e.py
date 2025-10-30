def remaining_area_after_years(H, W, Y, elevations):
    from collections import deque
    
    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize the results for each year
    results = []
    
    # Create a list of (elevation, position) and sort it
    elevation_positions = []
    for i in range(H):
        for j in range(W):
            elevation_positions.append((elevations[i][j], i, j))
    
    # Sort by elevation
    elevation_positions.sort()
    
    # Total area of the island
    total_area = H * W
    remaining_area = total_area
    
    # To keep track of which sections have sunk
    sunk = [[False] * W for _ in range(H)]
    
    # Queue for BFS
    queue = deque()
    
    # Current sea level
    current_sea_level = 0
    
    # Process each year
    for year in range(1, Y + 1):
        current_sea_level = year
        
        # Sink sections that are at or below the current sea level
        while elevation_positions and elevation_positions[0][0] <= current_sea_level:
            _, x, y = elevation_positions.pop(0)
            if not sunk[x][y]:
                # Start sinking process
                queue.append((x, y))
                sunk[x][y] = True
                remaining_area -= 1
        
        # BFS to sink adjacent sections
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not sunk[nx][ny] and elevations[nx][ny] <= current_sea_level:
                    sunk[nx][ny] = True
                    remaining_area -= 1
                    queue.append((nx, ny))
        
        # Store the result for this year
        results.append(remaining_area)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W, Y = map(int, data[0].split())
elevations = [list(map(int, line.split())) for line in data[1:H + 1]]

# Get the results
results = remaining_area_after_years(H, W, Y, elevations)

# Print the results
for result in results:
    print(result)