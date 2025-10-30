import sys

def solve():
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    N = len(S)
    M = len(T)

    # Early exit if the absolute difference in lengths is greater than K.
    # Each insert/delete operation changes length by 1.
    # Replacing doesn't change length.
    # So, at least |N - M| operations are needed to just match lengths.
    if abs(N - M) > K:
        print("No")
        return

    # dp array stores minimum operations for S[:i] to T[:j]
    # We use a band-limited DP, where `dp[d_idx]` represents the minimum operations
    # to transform S[:i] to T[:j], where `d = j - i` (difference in prefix lengths).
    # Since `d` can range from -K to K, we map it to `d_idx = d + K`.
    # So `d_idx` ranges from 0 to 2*K.
    
    # Initialize `dp` for i = 0 (empty S prefix).
    # `dp[d_idx]` refers to `dp[i-1][j]` or `dp[i-1][j-1]` from a conceptual full DP table.
    # For `i=0`, `S` is an empty string. `dist("", T[:j])` is `j` insertions.
    # Here `d = j - 0 = j`. So `d_idx = j + K`.
    # Valid `j` are from `0` to `K` (due to `abs(0-j) <= K`).
    
    dp = [float('inf')] * (2 * K + 1)
    for j_val in range(K + 1): # j_val goes from 0 to K
        # d_idx = j_val + K
        dp[j_val + K] = j_val

    # Iterate through S characters (from S[0] to S[N-1])
    # `i` represents the current length of S prefix (S[:i])
    for i in range(1, N + 1):
        new_dp = [float('inf')] * (2 * K + 1)

        # Handle the case where T prefix length is 0 (j=0).
        # `dist(S[:i], T[:0])` is `i` (deletions).
        # Here, `d = j - i = 0 - i = -i`.
        # `d_idx = K + d = K - i`.
        # This state is only relevant (i.e., within K operations) if `i <= K`.
        # If `K - i` is a valid index in `new_dp` array (i.e., `K-i >= 0`),
        # and if `i` is not already too large (i.e., `i <= K`), set its value.
        if K - i >= 0: # K-i is a valid d_idx
             # If i > K, then `dist(S[:i], "")` is `i` which is already greater than `K`,
             # so it can be considered `inf` for our purpose.
             # This means `new_dp[K-i]` should remain `inf` if `i > K`.
            if i <= K: 
                new_dp[K - i] = i

        # Iterate through relevant `d_idx` values for the current row `i`.
        # `d = j - i`, so `d_idx = (j - i) + K`.
        # `j` must be at least `1` (as `j=0` is handled above).
        # `j` must be at most `M`.
        # `j` must also be within `i-K` and `i+K` to be relevant.
        
        # Calculate the min and max `d_idx` to iterate.
        # min_j = max(1, i - K)
        # max_j = min(M, i + K)
        # min_d_idx = (min_j - i) + K
        # max_d_idx = (max_j - i) + K
        
        # d_idx range in `new_dp`:
        # Lower bound for d_idx: `max(0, K + 1 - i)` (since j must be >= 1)
        # Upper bound for d_idx: `min(2 * K, K + M - i)` (since j must be <= M)
        d_idx_start = max(0, K + 1 - i)
        d_idx_end = min(2 * K, K + M - i)

        for d_idx in range(d_idx_start, d_idx_end + 1):
            d = d_idx - K  # Current difference (j - i)
            j = i + d      # Corresponding column index in T

            # cost for match/replace (0 if chars match, 1 otherwise)
            char_match_cost = 0 if S[i-1] == T[j-1] else 1

            # Option 1: Substitution/Match. Depends on dp[i-1][j-1].
            # For cell (i, j), (i-1, j-1) has difference (j-1)-(i-1) = j-i = d.
            # So, it refers to `dp[d_idx]` from the previous row.
            val_sub = dp[d_idx] + char_match_cost
            
            # Option 2: Deletion. Depends on dp[i-1][j].
            # For cell (i, j), (i-1, j) has difference j-(i-1) = j-i+1 = d+1.
            # So, it refers to `dp[d_idx + 1]` from the previous row.
            val_del = float('inf')
            if d_idx + 1 <= 2 * K: # Check if d_idx + 1 is a valid index in `dp`
                val_del = dp[d_idx + 1] + 1
            
            # Option 3: Insertion. Depends on dp[i][j-1].
            # For cell (i, j), (i, j-1) has difference (j-1)-i = j-i-1 = d-1.
            # So, it refers to `new_dp[d_idx - 1]` (from the current row, already computed).
            val_ins = float('inf')
            if d_idx - 1 >= 0: # Check if d_idx - 1 is a valid index in `new_dp`
                val_ins = new_dp[d_idx - 1] + 1
            
            new_dp[d_idx] = min(val_sub, val_del, val_ins)
        
        # After computing all relevant cells for the current row, update dp to new_dp.
        dp = new_dp

    # The final result is the distance from S[:N] to T[:M].
    # The difference `d = M - N`.
    # The corresponding `d_idx` is `K + M - N`.
    final_d_idx = K + M - N
    
    # We've already checked `abs(N-M) > K` at the beginning, so `final_d_idx` will be a valid index.
    result = dp[final_d_idx]

    if result <= K:
        print("Yes")
    else:
        print("No")

solve()