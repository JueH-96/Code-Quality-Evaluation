def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    grid = data[1:]
    
    # Precompute the count of o's in each row and each column.
    row_count = [0] * N
    col_count = [0] * N
    for i in range(N):
        for j, ch in enumerate(grid[i]):
            if ch == 'o':
                row_count[i] += 1
                col_count[j] += 1
                
    # For every cell (i, j) with an o, it can be the intersection of
    # a pair of cells that share its row and another pair that share its column.
    # Such a triple is (i,j), (i, something) and (something, j). The counts
    # (excluding cell (i,j)) are (row_count[i] - 1) and (col_count[j] - 1).
    # Multiply these counts for each cell and sum over all cells.
    ans = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                ans += (row_count[i] - 1) * (col_count[j] - 1)
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()