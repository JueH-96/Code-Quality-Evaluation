def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read input parameters.
    it = iter(data)
    N = int(next(it))
    P = [int(next(it)) for _ in range(N)]
    Q = [int(next(it)) for _ in range(N)]
    
    # Build the inverse permutations.
    # invP[x] is the rank (0-indexed) of row number x.
    # invQ[x] is the rank (0-indexed) of column number x.
    invP = [0] * (N + 1)
    for rank, p in enumerate(P):
        invP[p] = rank
    invQ = [0] * (N + 1)
    for rank, q in enumerate(Q):
        invQ[q] = rank

    # We now construct the grid.
    # A useful construction is to put a '1' in cell (i,j) if and only if
    # invP[i] + invQ[j] >= N - 1; otherwise put '0'.
    # (Here i and j are 1-indexed representing row and column numbers.)
    #
    # It can be shown that with this assignment the row strings (S_i)
    # ordered by permutation P become lexicographically increasing. Similarly,
    # the column strings (T_j) ordered by permutation Q are lexicographically increasing.
    #
    # For example, consider the sample where N=3, P=[1,2,3] and Q=[2,1,3].
    # Then, invP = [unused, 0, 1, 2] and invQ = [unused, 1, 0, 2].
    # For row 1 (invP[1] = 0):
    #   - For column 1: 0+invQ[1]=0+1=1 (<2) → '0'
    #   - For column 2: 0+invQ[2]=0+0=0 (<2) → '0'
    #   - For column 3: 0+invQ[3]=0+2=2 (>=2) → '1'
    # with similar calculations for other rows.
    #
    # This method satisfies the conditions given in the problem.
    
    grid = []
    threshold = N - 1
    for i in range(1, N+1):
        row_rank = invP[i]
        row_chars = []
        for j in range(1, N+1):
            col_rank = invQ[j]
            if row_rank + col_rank >= threshold:
                row_chars.append("1")
            else:
                row_chars.append("0")
        grid.append("".join(row_chars))
    
    sys.stdout.write("
".join(grid))

if __name__ == '__main__':
    main()