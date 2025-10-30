import sys

MOD = 998244353

def solve():
    H, W = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))

    Q, sh, sw = map(int, sys.stdin.readline().split())
    # Adjust to 0-based indexing
    ch, cw = sh - 1, sw - 1

    # Calculate initial dp table
    # dp[h][w] = sum of products of values along paths from (0,0) to (h,w)
    dp = [[0 for _ in range(W)] for _ in range(H)]
    
    # Base case (0,0)
    dp[0][0] = A[0][0]
    
    # First row (h=0, w>0)
    for w in range(1, W):
        dp[0][w] = (dp[0][w-1] * A[0][w]) % MOD
        
    # First column (h>0, w=0)
    for h in range(1, H):
        dp[h][0] = (dp[h-1][0] * A[h][0]) % MOD
        
    # Rest of the table (h>0, w>0)
    for h in range(1, H):
        for w in range(1, W):
            dp[h][w] = (dp[h-1][w] + dp[h][w-1]) % MOD
            dp[h][w] = (dp[h][w] * A[h][w]) % MOD

    # Calculate initial S2prime table
    # S2prime[h][w] = sum of products along paths from (h,w) to (H-1, W-1), excluding A[h][w]
    S2prime = [[0 for _ in range(W)] for _ in range(H)]
    
    # Base case (H-1, W-1)
    S2prime[H-1][W-1] = 1
    
    # Last row (moving left, h=H-1, w<W-1)
    for w in range(W-2, -1, -1):
        # Path from (H-1, w) goes to (H-1, w+1). Product excludes A[H-1,w].
        # Sum of products from (H-1, w+1) to (H-1, W-1) excluding A[H-1,w+1] is S2prime[H-1,w+1]
        # The value A[H-1, w+1] is included in the product.
        S2prime[H-1][w] = (A[H-1][w+1] * S2prime[H-1][w+1]) % MOD
        
    # Last column (moving up, h<H-1, w=W-1)
    for h in range(H-2, -1, -1):
        # Path from (h, W-1) goes to (h+1, W-1). Product excludes A[h,W-1].
        # Sum of products from (h+1, W-1) to (H-1, W-1) excluding A[h+1,W-1] is S2prime[h+1,W-1]
        # The value A[h+1, W-1] is included in the product.
        S2prime[h][W-1] = (A[h+1][W-1] * S2prime[h+1][W-1]) % MOD

    # Rest of the table (moving up-left, h<H-1, w<W-1)
    for h in range(H-2, -1, -1):
        for w in range(W-2, -1, -1):
            # Paths from (h,w) go to (h+1,w) or (h,w+1). Product excludes A[h,w].
            # Sum of products from (h+1,w) onwards excluding A[h+1,w] is S2prime[h+1,w]. Include A[h+1,w].
            term_d = (A[h+1][w] * S2prime[h+1][w]) % MOD
            # Sum of products from (h,w+1) onwards excluding A[h,w+1] is S2prime[h,w+1]. Include A[h,w+1].
            term_r = (A[h][w+1] * S2prime[h][w+1]) % MOD
            S2prime[h][w] = (term_d + term_r) % MOD

    # Initial total sum
    current_total_sum = dp[H-1][W-1]

    # Process queries
    for _ in range(Q):
        d, a = sys.stdin.readline().split()
        a = int(a)

        # Determine the updated cell (h0, w0)
        h0, w0 = ch, cw
        if d == 'U':
            h0 -= 1
        elif d == 'D':
            h0 += 1
        elif d == 'L':
            w0 -= 1
        elif d == 'R':
            w0 += 1

        # Updated cell is (h0, w0)
        # Current position becomes the updated cell
        ch, cw = h0, w0

        A_old = A[h0][w0]
        A_new = a

        # Calculate dp[h0][w0] using old values
        # This value is needed for the contribution calculation before A[h0][w0] is updated
        if h0 == 0 and w0 == 0:
             # Special base case for (0,0): dp[0][0] is just A[0][0]
             dp_h0w0_old = A_old
        else:
             dp_sum_prev = 0
             if h0 > 0:
                 dp_sum_prev = (dp_sum_prev + dp[h0-1][w0]) % MOD
             if w0 > 0:
                 dp_sum_prev = (dp_sum_prev + dp[h0][w0-1]) % MOD
             dp_h0w0_old = (dp_sum_prev * A_old) % MOD

        # S2prime[h0][w0] value using old A values.
        # The formula for S2prime[h0][w0] doesn't depend on A[h0][w0],
        # only on A and S2prime values at cells (h>h0 or w>w0).
        # These dependent values are not changed by the update at (h0, w0),
        # so the stored S2prime[h0][w0] is correct for the 'old' state.
        S2prime_h0w0_old = S2prime[h0][w0]

        # Calculate the contribution factor C = dp[h0][w0]_old * S2prime[h0][w0]_old
        C = (dp_h0w0_old * S2prime_h0w0_old) % MOD

        # Calculate the change in total sum
        delta_A = (A_new - A_old + MOD) % MOD
        delta_S = (delta_A * C) % MOD

        # Update total sum
        current_total_sum = (current_total_sum + delta_S) % MOD

        # Update the grid value
        A[h0][w0] = A_new

        # Update the stored dp[h0][w0] value based on the new A[h0][w0]
        # This updated value will be used for future queries
        if h0 == 0 and w0 == 0:
             # Special base case for (0,0): dp[0][0] is just A[0][0]
             dp[h0][w0] = A[h0][w0]
        else:
             # dp[h0-1][w0] and dp[h0][w0-1] are the stored values, unaffected by this update
             dp_sum_prev_for_new = 0
             if h0 > 0:
                 dp_sum_prev_for_new = (dp_sum_prev_for_new + dp[h0-1][w0]) % MOD
             if w0 > 0:
                 dp_sum_prev_for_new = (dp_sum_prev_for_new + dp[h0][w0-1]) % MOD
             dp[h0][w0] = (dp_sum_prev_for_new * A[h0][w0]) % MOD

        # The stored S2prime[h0][w0] value does not need to be updated as its formula
        # does not depend on A[h0][w0]. Its neighbors' S2prime values and A values
        # also do not depend on A[h0][w0].

        # Print the current total sum
        print(current_total_sum)

solve()