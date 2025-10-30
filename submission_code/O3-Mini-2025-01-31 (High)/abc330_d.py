def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    grid = data[1:]
    
    # Precompute the number of 'o's in each row.
    row_count = [row.count('o') for row in grid]
    
    # Precompute the number of 'o's in each column.
    col_count = [0] * N
    for row in grid:
        for j, ch in enumerate(row):
            if ch == 'o':
                col_count[j] += 1

    # Here's the key observation:
    # A valid triple must span exactly two distinct rows and two distinct columns.
    # In any 2x2 block (formed by two selected rows and two selected columns),
    # exactly one cell (the intersection of the row with two cells and column with two cells)
    # acts as the common cell (pivot). The other two cells are chosen from the same row and same column as this pivot.
    #
    # Thus, if we fix a cell (r, c) with an 'o', and if there is at least one other 'o' in its row
    # and one other in its column, then this cell can be the pivot of a valid triple.
    # The number of triples for which (r, c) is the pivot is:
    #  (row_count[r] - 1) * (col_count[c] - 1)
    #
    # Every valid triple is counted exactly once with this method.
    
    ans = 0
    for i in range(N):
        # Only consider pivot cells in rows that have at least 2 'o's.
        r_extras = row_count[i] - 1
        if r_extras <= 0:
            continue
        for j, ch in enumerate(grid[i]):
            if ch == 'o' and col_count[j] > 1:
                ans += r_extras * (col_count[j] - 1)
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()