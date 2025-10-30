def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    grid = data[1:]
    
    # Precompute the number of rows that have an 'o' in each column.
    col_total = [0] * N
    for row in grid:
        for j, ch in enumerate(row):
            if ch == 'o':
                col_total[j] += 1

    ans = 0
    # The idea:
    # A valid triple consists of three 'o' cells such that
    #  - Two cells are in one row (say row r at columns c and d).
    #  - The third cell is in a different row,
    #    and shares its column with one of the two (either column c or d).
    #
    # For a given row r having L 'o's (in some columns),
    # let the indices be c1, c2, ..., cL.
    # For each pair (ci, cj), there are two possibilities:
    #   - Use the cell in another row in column ci. 
    #   - Use the cell in another row in column cj.
    # But the number of choices in column ci is (col_total[ci] - 1) (excluding the current row)
    # and similarly for column cj.
    # Thus, every pair in row r contributes:
    #   (col_total[ci] - 1) + (col_total[cj] - 1)
    # We can sum over pairs in r using a neat algebraic reformulation:
    #   Sum_{(ci, cj) with i < j} [(col_total[ci]-1) + (col_total[cj]-1)]
    # = (L-1) * ( Sum_{c in r with 'o'} (col_total[c]-1) ).
    #
    # Then, sum over each row to get the answer.
    #
    # The complexity is O(N^2) in the worst case (when grid is full of 'o'),
    # which is acceptable for N up to 2000.
    
    for row in grid:
        # Find all columns in this row with an 'o'
        cols = [j for j, ch in enumerate(row) if ch == 'o']
        L = len(cols)
        if L < 2:
            continue
        # Sum up (col_total[j]-1) for each column j in this row
        s = 0
        for j in cols:
            s += (col_total[j] - 1)
        # Each pair in the row contributes s multiplied by (L-1)
        ans += (L - 1) * s

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()