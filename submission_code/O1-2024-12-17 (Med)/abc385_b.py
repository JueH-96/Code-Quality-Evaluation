def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    H, W, X, Y = map(int, input_data[:4])
    # Next H lines are the grid
    grid = input_data[4:4+H]
    # The remainder is the movement string (T)
    T = input_data[4+H]
    
    # Convert to 0-based indexing
    x, y = X-1, Y-1
    
    # Keep track of visited house cells
    visited_houses = set()
    
    # A helper function to check if a cell is passable
    def is_passable(r, c):
        # We rely on the grid's boundary # to prevent out-of-bounds
        return grid[r][c] != '#'
    
    # Process each move
    for move in T:
        if move == 'U':
            if is_passable(x-1, y):
                x -= 1
        elif move == 'D':
            if is_passable(x+1, y):
                x += 1
        elif move == 'L':
            if is_passable(x, y-1):
                y -= 1
        elif move == 'R':
            if is_passable(x, y+1):
                y += 1
        
        # After moving, if it's a house, record it
        if grid[x][y] == '@':
            visited_houses.add((x, y))
    
    # Output the final position (convert back to 1-based) and number of distinct houses
    print(x+1, y+1, len(visited_houses))

# Do not forget to call main()
main()