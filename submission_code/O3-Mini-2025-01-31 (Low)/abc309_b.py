def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # First input: grid size N.
    N = int(input_data[0])
    
    grid = [list(input_data[i+1].strip()) for i in range(N)]
    
    # When N=2, a shift might have little effect, but even then should work
    # construct list of coordinates covering the outer boundary in order.
    coords = []
    
    # Top row (row 0, all columns)
    for j in range(0, N):
        coords.append((0, j))
    # Right column (rows from 1 to N-2), note not include corners
    for i in range(1, N-1):
        coords.append((i, N-1))
    # Bottom row (if distinct, row N-1, all columns in reverse order)
    if N > 1:
        for j in range(N-1, -1, -1):
            coords.append((N-1, j))
    # Left column (rows from N-2 to 1, in reverse order)
    for i in range(N-2, 0, -1):
        coords.append((i, 0))
        
    # Save the value from these coordinates in order.
    values = [grid[i][j] for (i, j) in coords]
    
    # Shift them clockwise by one.
    shifted = [values[-1]] + values[:-1]
    
    # Write back the values to grid.
    for idx, (i, j) in enumerate(coords):
        grid[i][j] = shifted[idx]
    
    # Print the grid.
    for row in grid:
        sys.stdout.write("".join(row) + "
")

if __name__ == '__main__':
    main()