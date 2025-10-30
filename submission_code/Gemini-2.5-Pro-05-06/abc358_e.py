import sys

def solve():
    K = int(sys.stdin.readline())
    C_values = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # Precompute combinations C(n, k) % MOD
    # pascal[n][k] will store C(n, k)
    pascal = [[0] * (K + 1) for _ in range(K + 1)]
    for i in range(K + 1):
        pascal[i][0] = 1
        if i == 0: # C(0,j) for j>0 is 0, C(0,0)=1, already set
            continue
        # C(i,j) for j from 1 to i
        for j in range(1, i + 1): 
            pascal[i][j] = (pascal[i-1][j-1] + pascal[i-1][j]) % MOD

    # dp[j] stores number of ways to make string of length j using chars processed so far
    # Initialize for 0 characters processed.
    dp = [0] * (K + 1)
    dp[0] = 1 # Base case: 1 way to make empty string (use 0 of any char type)

    for char_idx in range(26): # Process 26 letters A-Z
        limit_char_count = C_values[char_idx]
        
        # Optimization: if current char cannot be used (limit is 0),
        # then we can only add 0 instances of it.
        # This means dp_next_state[len] = dp[len] * C(len,0) = dp[len].
        # So the dp table remains unchanged.
        if limit_char_count == 0:
            continue

        dp_next_state = [0] * (K + 1)
        for prev_len in range(K + 1): # length of string using previous char types
            if dp[prev_len] == 0: # if no way to form string of prev_len, skip
                continue
            
            # Value from dp table for strings made of previous char types
            val_dp_prev_len = dp[prev_len] 
            
            # Iterate over number of current character type to add.
            # Max items that can be added is limit_char_count.
            # Also, total length cannot exceed K, so max items to add is K - prev_len.
            max_items_to_add_due_to_K = K - prev_len
            
            # Iterate num_added from 0 up to min(limit_char_count, max_items_to_add_due_to_K)
            for num_added in range(min(limit_char_count, max_items_to_add_due_to_K) + 1):
                current_total_len = prev_len + num_added
                
                # Ways to choose positions for num_added items from current_total_len positions
                combinations = pascal[current_total_len][num_added]
                
                term = (val_dp_prev_len * combinations) % MOD
                
                dp_next_state[current_total_len] = \
                    (dp_next_state[current_total_len] + term) % MOD
        
        # Current character processed, update dp table for next iteration
        dp = dp_next_state

    ans = 0
    # Summing up counts for strings of length 1 to K (inclusive)
    for length in range(1, K + 1):
        ans = (ans + dp[length]) % MOD
    
    sys.stdout.write(str(ans) + "
")

solve()