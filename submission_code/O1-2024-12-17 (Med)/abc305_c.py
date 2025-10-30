def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid_strings = input_data[2:]
    
    # Read the grid into a 2D list
    grid = [list(row) for row in grid_strings]
    
    # Initialize bounding box for '#' squares
    min_row, max_row = H, -1
    min_col, max_col = W, -1
    
    # Find bounding box for all '#' squares
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < min_row: min_row = i
                if i > max_row: max_row = i
                if j < min_col: min_col = j
                if j > max_col: max_col = j

    # Within that bounding box, find the single '.' that must be the missing cookie
    # The problem guarantees there is exactly one such square
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                # Print 1-based indices
                print(i+1, j+1)
                return

# Do not forget to call main()
main()