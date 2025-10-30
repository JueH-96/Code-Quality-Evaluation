def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[i + 2] for i in range(H)]
    
    active_rows = set(range(H))
    active_cols = set(range(W))
    
    while True:
        rows_to_remove = set()
        # Check rows
        for row in active_rows:
            colors = set(grid[row][j] for j in active_cols)
            if len(colors) == 1 and len(active_cols) >= 2:
                rows_to_remove.add(row)
        
        cols_to_remove = set()
        # Check columns
        for col in active_cols:
            colors = set(grid[i][col] for i in active_rows)
            if len(colors) == 1 and len(active_rows) >= 2:
                cols_to_remove.add(col)
        
        # Remove marked rows and columns
        if not rows_to_remove and not cols_to_remove:
            break
        active_rows -= rows_to_remove
        active_cols -= cols_to_remove
    
    # Number of remaining cookies is the product of remaining rows and columns
    print(len(active_rows) * len(active_cols))

if __name__ == "__main__":
    main()