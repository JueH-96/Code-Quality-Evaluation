def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W = map(int, data[:2])
    grid = data[2:]
    
    # Initialize boundaries for the smallest and largest row/col where '#' is found
    row_min, row_max = H, 0
    col_min, col_max = W, 0
    
    # Find the bounding rectangle of all '#' cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                row_min = min(row_min, i)
                row_max = max(row_max, i)
                col_min = min(col_min, j)
                col_max = max(col_max, j)
    
    # Within that bounding rectangle, find the one cell that is '.' 
    # (that must be the cookie Snuke ate)
    for i in range(row_min, row_max + 1):
        for j in range(col_min, col_max + 1):
            if grid[i][j] == '.':
                print(i+1, j+1)
                return

# Do not forget to call main()!
if __name__ == "__main__":
    main()