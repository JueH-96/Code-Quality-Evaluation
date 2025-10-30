def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def count_humidified_cells(H, W, D, grid, pos1, pos2):
    humidified = set()
    
    # For each cell in the grid
    for i in range(H):
        for j in range(W):
            # Skip if it's a desk
            if grid[i][j] == '#':
                continue
            
            # Check if this cell is within distance D of either humidifier
            dist1 = manhattan_distance(i, j, pos1[0], pos1[1])
            dist2 = manhattan_distance(i, j, pos2[0], pos2[1])
            
            if dist1 <= D or dist2 <= D:
                humidified.add((i, j))
    
    return len(humidified)

def solve():
    # Read input
    H, W, D = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    
    # Get all floor positions
    floor_positions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_positions.append((i, j))
    
    # Try all possible pairs of positions for humidifiers
    max_humidified = 0
    for i in range(len(floor_positions)):
        for j in range(i + 1, len(floor_positions)):
            pos1 = floor_positions[i]
            pos2 = floor_positions[j]
            humidified = count_humidified_cells(H, W, D, grid, pos1, pos2)
            max_humidified = max(max_humidified, humidified)
    
    print(max_humidified)

# Run the solution
solve()