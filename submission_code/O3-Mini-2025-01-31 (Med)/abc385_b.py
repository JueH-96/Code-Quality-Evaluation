def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    # First line: H, W, X, Y
    first_line = input_data[0].split()
    H = int(first_line[0])
    W = int(first_line[1])
    X = int(first_line[2])
    Y = int(first_line[3])
    
    # Read grid: next H lines
    grid = []
    for i in range(1, H+1):
        grid.append(input_data[i])
    
    # T is the next line
    T = input_data[H+1].strip()
    
    # Using 0-indexed internally, but conversion needed
    x = X - 1
    y = Y - 1
    
    # Define move dictionary
    moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    
    # Set to keep track of visited houses
    visited_houses = set()
    
    # Function to check if a cell is passable
    def is_passable(nx, ny):
        # It is passable if within bounds and not '#'
        if 0 <= nx < H and 0 <= ny < W:
            return grid[nx][ny] != '#'
        return False
    
    for instr in T:
        dx, dy = moves[instr]
        nx, ny = x + dx, y + dy
        if is_passable(nx, ny):
            x, y = nx, ny
            # If landed or passed through, check for house marker '@'
            if grid[x][y] == '@':
                visited_houses.add((x, y))
    
    # Return final output in 1-indexed form.
    print(x+1, y+1, len(visited_houses))
    
if __name__ == '__main__':
    main()