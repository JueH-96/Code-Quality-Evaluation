def main():
    import sys
    # Use sys.stdin.buffer for fast input reading.
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Create an iterator over the tokens.
    it = iter(data)
    # Read N and Q
    N = int(next(it))
    Q = int(next(it))
    # Read the N rows of the pattern.
    pattern = [next(it).decode() for _ in range(N)]
    
    # Precompute the 2D prefix sum for the N x N pattern.
    # We'll build a (N+1)x(N+1) array "pref" where:
    #   pref[i+1][j+1] = sum{ grid[x][y] for x in 0..i, y in 0..j }
    # Here, grid[i][j] = 1 if pattern[i][j]=='B' else 0.
    pref = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        row = pattern[i]
        for j in range(N):
            val = 1 if row[j] == 'B' else 0
            pref[i+1][j+1] = pref[i][j+1] + pref[i+1][j] - pref[i][j] + val
    tot = pref[N][N]  # total black in one full period.

    # F(x, y) returns the number of black cells in the rectangle from (0,0) to (x,y) inclusive.
    def F(x, y):
        if x < 0 or y < 0:
            return 0
        # Number of rows and columns in region [0, x] x [0, y]
        R = x + 1
        C = y + 1
        # Divide into complete pattern blocks and remainders.
        a, r = divmod(R, N)
        b, c = divmod(C, N)
        # Count black squares:
        #   - a*b complete blocks, each with tot black cells.
        #   - For the a complete blocks (full vertical blocks) in the column remainder c, add pref[N][c].
        #   - For the b complete blocks (full horizontal blocks) in the row remainder r, add pref[r][N].
        #   - And add the leftover part at the bottom-right: pref[r][c].
        return a * b * tot + a * pref[N][c] + b * pref[r][N] + pref[r][c]
    
    # Process each query and use inclusion-exclusion:
    # For a query given by A, B, C, D, where (A, B) is top-left and (C, D) is bottom-right,
    # the answer equals: F(C, D) - F(A-1, D) - F(C, B-1) + F(A-1, B-1)
    out_lines = []
    for _ in range(Q):
        A = int(next(it))
        B = int(next(it))
        C = int(next(it))
        D = int(next(it))
        ans = F(C, D) - F(A - 1, D) - F(C, B - 1) + F(A - 1, B - 1)
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()