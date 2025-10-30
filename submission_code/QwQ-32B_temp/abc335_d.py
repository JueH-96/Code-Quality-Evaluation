def main():
    import sys
    N = int(sys.stdin.readline())
    center = (N - 1) // 2
    path = []
    for l in range(N // 2):
        row_start = l
        row_end = N - 1 - l
        col_start = l
        col_end = N - 1 - l

        # Top row left to right
        for col in range(col_start, col_end + 1):
            path.append((row_start, col))
        
        # Right column from row_start+1 to row_end
        for row in range(row_start + 1, row_end + 1):
            path.append((row, col_end))
        
        # Bottom row from col_end-1 to col_start if needed
        if row_start < row_end:
            for col in range(col_end - 1, col_start - 1, -1):
                path.append((row_end, col))
        
        # Left column from row_end-1 down to row_start+1 if needed
        if col_start < col_end:
            for row in range(row_end - 1, row_start, -1):
                path.append((row, col_start))
    
    # Initialize grid
    grid = [['0' for _ in range(N)] for _ in range(N)]
    grid[center][center] = 'T'
    for idx, (i, j) in enumerate(path):
        grid[i][j] = str(idx + 1)
    
    # Print each row
    for row in grid:
        print(' '.join(row))

if __name__ == "__main__":
    main()