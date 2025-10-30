import sys

def solve():
    """
    Main function to solve the problem.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        H, W = map(int, readline().split())
        S_orig = [readline().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    MOD = 998244353

    # Transpose if H > W to keep H the smaller dimension
    if H > W:
        S = [['' for _ in range(H)] for _ in range(W)]
        for r in range(H):
            for c in range(W):
                S[c][r] = S_orig[r][c]
        H, W = W, H
    else:
        S = [list(row) for row in S_orig]

    # Convert grid characters to numeric values {0, 1, 2}
    grid_num = [[-1] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if S[r][c] != '?':
                grid_num[r][c] = int(S[r][c]) - 1

    # Precompute powers of 3 for mask manipulation
    pow3 = [1] * (H + 1)
    for i in range(1, H + 1):
        pow3[i] = pow3[i-1] * 3
    
    MAX_MASKS = pow3[H]

    # Precompute masks that are vertically valid
    vert_valid_masks = []
    if H == 1:
        vert_valid_masks = list(range(MAX_MASKS))
    else:
        for mask in range(MAX_MASKS):
            is_vert_ok = True
            for r in range(H - 1):
                d1 = (mask // pow3[r]) % 3
                d2 = (mask // pow3[r+1]) % 3
                if d1 == d2:
                    is_vert_ok = False
                    break
            if is_vert_ok:
                vert_valid_masks.append(mask)

    # Precompute masks that are valid for each column (vertical + grid constraints)
    is_mask_valid_for_col = [[False] * MAX_MASKS for _ in range(W)]
    for c in range(W):
        for mask in vert_valid_masks:
            is_compat_ok = True
            for r in range(H):
                d = (mask // pow3[r]) % 3
                s_val = grid_num[r][c]
                if s_val != -1 and d != s_val:
                    is_compat_ok = False
                    break
            if is_compat_ok:
                is_mask_valid_for_col[c][mask] = True

    # Initialize DP for the first column
    dp = [0] * MAX_MASKS
    for mask in range(MAX_MASKS):
        if is_mask_valid_for_col[0][mask]:
            dp[mask] = 1

    # Buffers for the transform
    dp_in = [0] * MAX_MASKS
    dp_out = [0] * MAX_MASKS
    
    # Main DP loop over columns
    for c in range(1, W):
        dp_in, dp_out = dp, dp_out # dp_in holds current state
        
        # Fast transform (Sum over non-equal indices)
        for k in range(H):
            p = pow3[k]
            for j in range(MAX_MASKS):
                if (j // p) % 3 == 0:
                    m0, m1, m2 = j, j + p, j + 2*p
                    v0, v1, v2 = dp_in[m0], dp_in[m1], dp_in[m2]
                    s = (v0 + v1 + v2) % MOD
                    dp_out[m0] = (s - v0 + MOD) % MOD
                    dp_out[m1] = (s - v1 + MOD) % MOD
                    dp_out[m2] = (s - v2 + MOD) % MOD
            dp_in, dp_out = dp_out, dp_in # Swap buffers
        
        transformed_dp = dp_in
        
        # Update DP table for the current column by filtering with valid masks
        new_dp = [0] * MAX_MASKS
        for mask in vert_valid_masks:
            if is_mask_valid_for_col[c][mask]:
                new_dp[mask] = transformed_dp[mask]
        
        dp = new_dp
    
    # The answer is the sum of ways for the last column
    ans = sum(dp) % MOD
    print(ans)

solve()