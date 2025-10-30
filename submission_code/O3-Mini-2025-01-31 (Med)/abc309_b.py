def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    grid = [list(line.strip()) for line in input_data[1:1+N]]
    
    if N < 2:
        # For safety although constraint N>=2.
        for row in grid:
            print("".join(row))
        return
    
    # Extract the outer border cells in clockwise order starting from (0,0)
    border = []
    # top row (left to right)
    for j in range(N):
        border.append((0, j))
    # right column (top to bottom, excluding the top corner)
    for i in range(1, N):
        border.append((i, N-1))
    # bottom row (right to left, excluding the bottom right corner)
    for j in range(N-2, -1, -1):
        border.append((N-1, j))
    # left column (bottom to top, excluding both corners)
    for i in range(N-2, 0, -1):
        border.append((i, 0))
    
    # Get the values in this border order
    border_values = [grid[i][j] for i, j in border]
    
    # Shift the border values clockwise by one means each cell takes the value of its predecessor.
    # So the first cell gets the value of the last cell, and others get the previous.
    rotated = [border_values[-1]] + border_values[:-1]
    
    # Place the rotated values back into the grid.
    for idx, (i, j) in enumerate(border):
        grid[i][j] = rotated[idx]
    
    # Print the modified grid.
    for row in grid:
        sys.stdout.write("".join(row) + "
")

if __name__ == '__main__':
    main()