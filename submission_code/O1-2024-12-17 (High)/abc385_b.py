def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    # Parse the first line: H, W, X, Y
    H, W, start_x, start_y = map(int, data[:4])
    
    # Next H lines are the grid
    grid = data[4:4+H]
    
    # The next token is the movement string T
    T = data[4+H]
    
    # Convert to zero-based indices
    x, y = start_x - 1, start_y - 1

    # A set to record visited houses
    visited_houses = set()
    
    # Define moves
    moves = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }
    
    # Simulate the moves
    for char in T:
        dx, dy = moves[char]
        nx, ny = x + dx, y + dy
        
        # Check passability (impassable = '#')
        if grid[nx][ny] != '#':
            x, y = nx, ny
        
        # If it's a house, record it
        if grid[x][y] == '@':
            visited_houses.add((x, y))
    
    # Print final position (converted back to 1-based) and count of distinct houses visited
    print(x+1, y+1, len(visited_houses))

# Do not forget to call main()
if __name__ == "__main__":
    main()