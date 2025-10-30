def solve_santa_path():
    # Read input
    H, W, X, Y = map(int, input().split())
    grid = [input() for _ in range(H)]
    T = input().strip()
    
    # Current position (convert from 1-indexed to 0-indexed for internal calculations)
    x, y = X - 1, Y - 1
    
    # Track houses visited
    houses_visited = set()
    
    # Process each movement instruction
    for move in T:
        # Calculate potential new position
        new_x, new_y = x, y
        
        if move == 'U':
            new_x = x - 1
        elif move == 'D':
            new_x = x + 1
        elif move == 'L':
            new_y = y - 1
        elif move == 'R':
            new_y = y + 1
        
        # Check if new position is passable
        if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
            x, y = new_x, new_y
            
            # Check if current position has a house
            if grid[x][y] == '@':
                houses_visited.add((x, y))
    
    # Convert back to 1-indexed for the output
    return f"{x + 1} {y + 1} {len(houses_visited)}"

# Main program
print(solve_santa_path())