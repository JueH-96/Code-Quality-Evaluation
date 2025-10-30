def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    
    # Read pattern and convert to integer grid (B = 1, W = 0)
    pattern = []
    for _ in range(N):
        s = next(it).decode().strip()
        row = [1 if ch == 'B' else 0 for ch in s]
        pattern.append(row)
    
    # Build 2D prefix sum for the N x N pattern.
    # PP[i][j] stores the sum for the submatrix pattern[0:i][0:j]
    PP = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        row_sum = 0
        for j in range(1, N+1):
            row_sum += pattern[i-1][j-1]
            PP[i][j] = PP[i-1][j] + row_sum

    # total black count in the complete pattern.
    total_pattern = PP[N][N]

    # Define f(x, y): given x,y (with x,y >=0) return the number of black squares
    # in the rectangle from (0,0) to (x,y) inclusive in the infinite grid.
    #
    # Note that the grid is built by tiling the pattern in both dimensions.
    # Let R = x+1 and C = y+1 (number of rows and columns in the rectangle).
    # Express R = a*N + r and C = b*N + c, with 0 <= r, c < N.
    # Then the rectangle splits into:
    #   1. a*b complete blocks each with total_pattern blacks.
    #   2. a vertical strip of full-pattern-rows with c columns:
    #         contributes a * PP[N][c]
    #   3. b horizontal strips of r pattern rows (all columns):
    #         contributes b * PP[r][N]
    #   4. A bottom-right leftover block of r rows and c columns:
    #         contributes PP[r][c]
    def f(x, y):
        if x < 0 or y < 0:
            return 0
        R = x + 1
        C = y + 1
        a, r = divmod(R, N)
        b, c = divmod(C, N)
        return a * b * total_pattern + a * PP[N][c] + b * PP[r][N] + PP[r][c]

    # Use inclusion-exclusion to answer each query.
    # For a query (A, B, C, D), the number of blacks in the rectangle with top-left (A,B)
    # and bottom-right (C,D) is:
    #   f(C, D) - f(A-1, D) - f(C, B-1) + f(A-1, B-1)
    out_lines = []
    for _ in range(Q):
        A = int(next(it))
        B = int(next(it))
        C = int(next(it))
        D = int(next(it))
        ans = f(C, D) - f(A-1, D) - f(C, B-1) + f(A-1, B-1)
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()