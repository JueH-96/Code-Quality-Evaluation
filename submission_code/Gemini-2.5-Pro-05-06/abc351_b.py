def solve():
    # Read the size of the grids
    N = int(input())

    # Read grid A
    # grid_a will be a list of N strings, where each string is a row
    grid_a = []
    for _ in range(N):
        grid_a.append(input())

    # Read grid B
    # grid_b will be a list of N strings
    grid_b = []
    for _ in range(N):
        grid_b.append(input())

    # Iterate through the grids to find the differing cell
    for r in range(N):  # r is the 0-indexed row number (0 to N-1)
        for c in range(N):  # c is the 0-indexed column number (0 to N-1)
            # Compare characters at the current cell (r, c)
            if grid_a[r][c] != grid_b[r][c]:
                # Found the differing cell.
                # The problem asks for 1-indexed coordinates.
                row_1_indexed = r + 1
                col_1_indexed = c + 1
                
                # Print the 1-indexed row and column, space-separated
                print(f"{row_1_indexed} {col_1_indexed}")
                
                # Since there is exactly one differing cell,
                # we can stop the program after finding it.
                return

# Call the solve function to execute the logic
if __name__ == '__main__':
    solve()