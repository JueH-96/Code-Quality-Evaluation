import sys

# Set recursion limit for large inputs, although not strictly needed for iterative DP
# but good practice for competitive programming with potential large recursion.
sys.setrecursionlimit(10**6)

def solve():
    H, W = map(int, sys.stdin.readline().split())
    
    # Read initial grid A (0-indexed internally)
    initial_A = []
    for _ in range(H):
        initial_A.append(list(map(int, sys.stdin.readline().split())))
    
    Q_val = int(sys.stdin.readline())
    sh, sw = map(int, sys.stdin.readline().split()) # Takahashi's start position (1-indexed)
    
    queries = []
    for _ in range(Q_val):
        d, a = sys.stdin.readline().split()
        queries.append((d, int(a)))

    MOD = 998244353

    # Internal grid A (mutable copy)
    A = [row[:] for row in initial_A]
    
    # Adjust sh, sw to be 0-indexed internally for Takahashi's current position
    current_sh = sh - 1
    current_sw = sw - 1

    # DP table for paths from (0,0) to (h,w) (inclusive A[h][w])
    # dp_f[h][w] = sum of products for paths from (0,0) to (h,w)
    dp_f = [[0] * W for _ in range(H)]
    
    # DP table for paths from (h,w) to (H-1,W-1) (inclusive A[h][w])
    # dp_b[h][w] = sum of products for paths from (h,w) to (H-1,W-1)
    dp_b = [[0] * W for _ in range(H)]

    # Function to compute/recompute dp_f table
    # This must be called for the affected region (h_start to H-1, w_start to W-1)
    def recompute_dp_f(h_start, w_start):
        for h in range(h_start, H):
            for w in range(w_start, W):
                if h == 0 and w == 0:
                    dp_f[0][0] = A[0][0]
                    continue
                
                val_from_up = 0
                if h > 0:
                    val_from_up = dp_f[h-1][w]
                
                val_from_left = 0
                if w > 0:
                    val_from_left = dp_f[h][w-1]
                
                dp_f[h][w] = (val_from_up + val_from_left) * A[h][w] % MOD

    # Function to compute/recompute dp_b table
    # This must be called for the affected region (h_start down to 0, w_start down to 0)
    def recompute_dp_b(h_start, w_start):
        for h in range(h_start, -1, -1):
            for w in range(w_start, -1, -1):
                if h == H-1 and w == W-1:
                    dp_b[H-1][W-1] = A[H-1][W-1]
                    continue
                
                val_from_down = 0
                if h < H - 1:
                    val_from_down = dp_b[h+1][w]
                
                val_from_right = 0
                if w < W - 1:
                    val_from_right = dp_b[h][w+1]
                
                dp_b[h][w] = (val_from_down + val_from_right) * A[h][w] % MOD

    # Initial computations of full DP tables
    recompute_dp_f(0, 0)
    recompute_dp_b(0, 0)
    
    current_total_sum = dp_f[H-1][W-1] # The total sum for the initial grid

    for d, a in queries:
        # Determine the new position of Takahashi (r, c)
        if d == 'L': current_sw -= 1
        elif d == 'R': current_sw += 1
        elif d == 'U': current_sh -= 1
        elif d == 'D': current_sh += 1
        
        # (r,c) is the cell that is being updated
        r, c = current_sh, current_sw
        
        old_val_rc = A[r][c] # Value in A[r][c] before update
        new_val_rc = a     # New value for A[r][c]

        # Calculate factor_prefix (sum of products for paths from (0,0) to (r,c) excluding A[r][c])
        factor_prefix = 0
        if r == 0 and c == 0:
            factor_prefix = 1 # For (0,0), the prefix product excluding A[0][0] is 1
        elif r == 0: # on first row (h=0), not (0,0)
            factor_prefix = dp_f[r][c-1]
        elif c == 0: # on first column (w=0), not (0,0)
            factor_prefix = dp_f[r-1][c]
        else: # general case
            factor_prefix = (dp_f[r-1][c] + dp_f[r][c-1]) % MOD
        
        # Calculate factor_suffix (sum of products for paths from (r,c) to (H-1,W-1) excluding A[r][c])
        factor_suffix = 0
        if r == H-1 and c == W-1:
            factor_suffix = 1 # For (H-1,W-1), the suffix product excluding A[H-1][W-1] is 1
        elif r == H-1: # on last row (h=H-1), not (H-1,W-1)
            factor_suffix = dp_b[r][c+1]
        elif c == W-1: # on last column (w=W-1), not (H-1,W-1)
            factor_suffix = dp_b[r+1][c]
        else: # general case
            factor_suffix = (dp_b[r+1][c] + dp_b[r][c+1]) % MOD
        
        # Update current_total_sum using the differential approach
        diff_val = (new_val_rc - old_val_rc + MOD) % MOD
        change_in_sum = diff_val * factor_prefix % MOD * factor_suffix % MOD
        current_total_sum = (current_total_sum + change_in_sum) % MOD
        
        # Print the answer for the current query
        sys.stdout.write(str(current_total_sum) + '
')
        
        # Update the grid cell A[r][c] to its new value
        A[r][c] = new_val_rc

        # Recompute affected parts of dp_f and dp_b tables
        # dp_f: affected cells are (h, w) where h >= r and w >= c
        recompute_dp_f(r, c)
        
        # dp_b: affected cells are (h, w) where h <= r and w <= c
        recompute_dp_b(r, c)

# Call the solver function
solve()