import sys

def solve():
    K_limit = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()
    T_str = sys.stdin.readline().strip()

    N = len(S_str)
    M = len(T_str)

    if abs(N - M) > K_limit:
        print("No")
        return

    # dp_prev stores costs for row i-1, dp_curr for row i.
    # dp_array[k_idx] stores dp[current_row_idx][j] 
    # where j - current_row_idx = k = k_idx - K_limit
    # Array size: 2*K_limit + 1. k_idx ranges from 0 to 2*K_limit.
    # k (offset j-i) ranges from -K_limit to K_limit.
    
    # Initialize dp_prev for row 0 (dp[0][...])
    # dp[0][j] = j (cost of inserting j characters from T_str)
    # For dp[0][j], i=0, so k = j - 0 = j. k_idx = K_limit + j.
    # We need 0 <= j <= K_limit and j <= M.
    dp_prev = [K_limit + 1] * (2 * K_limit + 1)
    for j in range(min(K_limit, M) + 1):
        k_idx = K_limit + j
        dp_prev[k_idx] = j

    # Iterate i from 1 to N (row index for DP table, S_str[i-1] is the character)
    for i in range(1, N + 1):
        dp_curr = [K_limit + 1] * (2 * K_limit + 1)
        
        # Determine the valid range of k_idx for this row i.
        # k = j - i.
        # Smallest j is 0 => smallest k is -i => smallest k_idx is K_limit - i.
        # Largest j is M  => largest k is M-i => largest k_idx is K_limit + M - i.
        # Combined with band constraint (0 <= k_idx <= 2*K_limit):
        min_k_idx_for_row_i = max(0, K_limit - i)
        max_k_idx_for_row_i = min(2 * K_limit, K_limit + M - i)

        for k_idx in range(min_k_idx_for_row_i, max_k_idx_for_row_i + 1):
            k = k_idx - K_limit
            j = i + k # Current column in T_str for cell dp[i][j]
            # Due to min/max_k_idx_for_row_i calculation, j is always in [0, M].

            val_match_mismatch = K_limit + 1
            val_delete = K_limit + 1
            val_insert = K_limit + 1

            # Option 1: Match/Mismatch from dp[i-1][j-1]
            # Needs S_str[i-1] and T_str[j-1]. So i > 0 (true by loop) and j > 0.
            # Previous cell (i-1,j-1) has diagonal (j-1)-(i-1) = j-i = k. Same k_idx.
            if j > 0: # T_str[j-1] exists
                cost = 1 if S_str[i-1] != T_str[j-1] else 0
                if dp_prev[k_idx] <= K_limit : 
                     val_match_mismatch = dp_prev[k_idx] + cost
            
            # Option 2: Deletion from S_str (from dp[i-1][j])
            # Needs S_str[i-1]. Previous cell (i-1,j) has diagonal j-(i-1) = j-i+1 = k+1.
            # So, k_idx for dp_prev is k_idx + 1.
            if k_idx + 1 <= 2 * K_limit : # Ensure k_idx+1 is a valid index
                if dp_prev[k_idx+1] <= K_limit: # Check if source value is not "infinity"
                    val_delete = dp_prev[k_idx+1] + 1
            
            # Option 3: Insertion into S_str (from dp[i][j-1])
            # Needs T_str[j-1]. So j > 0.
            # Previous cell (i,j-1) has diagonal (j-1)-i = k-1.
            # So, k_idx for dp_curr is k_idx - 1.
            if j > 0: # T_str[j-1] exists
                if k_idx - 1 >= 0: # Ensure k_idx-1 is a valid index
                    if dp_curr[k_idx-1] <= K_limit: # Check if source value is not "infinity"
                         val_insert = dp_curr[k_idx-1] + 1
            
            dp_curr[k_idx] = min(val_match_mismatch, val_delete, val_insert)
            
        dp_prev = dp_curr # Python list assignment creates a reference; need a copy if dp_curr is modified later.
                          # Here, dp_curr is recreated each iteration, so dp_prev = dp_curr is fine to carry over.

    # Final answer is dp[N][M]
    # For dp[N][M]: k = M - N. k_idx = K_limit + M - N.
    target_k_idx = K_limit + M - N
    
    # abs(N-M) <= K_limit check at start ensures target_k_idx is in [0, 2*K_limit]
    final_val = dp_prev[target_k_idx]

    if final_val <= K_limit:
        print("Yes")
    else:
        print("No")

solve()