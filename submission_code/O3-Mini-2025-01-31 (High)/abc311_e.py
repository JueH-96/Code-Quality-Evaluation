def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Parse the first three numbers: H, W, N.
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    N = int(next(it))
    
    # Create the grid as a list of bytearrays.
    # A cell will be 0 if it is not holed and 1 if holed.
    # Using bytearray is memory‐efficient for storing 0/1 values.
    grid = [bytearray(W) for _ in range(H)]
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        grid[a - 1][b - 1] = 1  # Mark the holed square (convert 1-indexed into 0-indexed)
    
    # We will use dynamic programming to count the number of holeless squares.
    # Let dp[i][j] be the side length of the largest square (all holeless)
    # that ends at cell (i,j) (1-indexed in our dp arrays).
    # Then every square of side length k (1 <= k <= dp[i][j]) ending at (i,j)
    # is a valid holeless square.
    # We use only two rows in our dp computation.
    dp_prev = [0] * (W + 1)
    total = 0
    for r in range(H):
        dp_cur = [0] * (W + 1)
        # Save a local reference to the current grid row.
        row = grid[r]
        # Use local names for the dp_prev and dp_cur lists to help micro-opt.
        prev = dp_prev
        cur = dp_cur
        for j in range(1, W + 1):
            if row[j - 1]:
                # There is a hole at this cell.
                cur[j] = 0
            else:
                # The cell is not holed.
                # dp_cur[j] = 1 + min(dp_prev[j], dp_cur[j-1], dp_prev[j-1])
                a = prev[j]
                b = cur[j - 1]
                if b < a:
                    a = b
                c = prev[j - 1]
                if c < a:
                    a = c
                cur[j] = a + 1
            total += cur[j]
        dp_prev = dp_cur  # Move to next row.
        
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()