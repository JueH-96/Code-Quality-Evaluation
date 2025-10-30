def main():
    import sys
    grid = [line.strip() for line in sys.stdin.readlines() if line.strip()]
    
    # Determine the safe rows: rows with no existing pieces.
    safe_rows = []
    for i in range(8):
        if '#' not in grid[i]:
            safe_rows.append(i)
    
    # Determine the safe columns: columns with no existing pieces.
    safe_cols = []
    for j in range(8):
        safe = True
        for i in range(8):
            if grid[i][j] == '#':
                safe = False
                break
        if safe:
            safe_cols.append(j)
    
    # The square (i,j) is safe if row i is safe and column j is safe.
    count = len(safe_rows) * len(safe_cols)
    print(count)

if __name__ == '__main__':
    main()