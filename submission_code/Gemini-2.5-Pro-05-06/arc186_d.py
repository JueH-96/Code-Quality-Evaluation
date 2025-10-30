import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    count_dp = [[0] * (N + 1) for _ in range(N + 1)]

    if N == 0:
        print(0)
        return

    count_dp[0][0] = 1

    # Precompute count_dp table
    # count_dp[k_len][bal] = sum count_dp[k_len-1][p]
    # where p ranges from max(1, bal-1) to k_len-1 (for k_len-1 > 0)
    # or p=0 (for k_len-1 = 0)
    
    # row_prefix_sum_for_prev_len[p_idx] = sum_{j=0..p_idx} count_dp[k_len-1][j]
    row_prefix_sum_for_prev_len = [0] * (N + 1) 

    # Base for k_len=1: prev_len = 0. count_dp[0][0]=1.
    row_prefix_sum_for_prev_len[0] = count_dp[0][0]
    # All other count_dp[0][j] are 0, so prefix sums don't change for j>0 for count_dp[0] row.

    for k_len in range(1, N + 1):
        # Calculate count_dp[k_len][bal]
        for bal in range(1, k_len + 1):
            if k_len - 1 == 0: # prev_len is 0
                # prev_bal must be 0. current_bal + val - 1 = 0.
                # Here, bal + val - 1 = 0. (using problem's `bal` as `current_bal` for this subproblem)
                # val = 0 is the only option for count_dp[0][0].
                # So, bal + 0 - 1 = 0  => bal = 1.
                if bal == 1:
                    count_dp[k_len][bal] = count_dp[0][0] 
                else:
                    count_dp[k_len][bal] = 0
            else: # prev_len > 0. prev_bal must be > 0.
                # Sum count_dp[k_len-1][p] for p from p_start_idx to p_end_idx
                p_start_idx = bal - 1
                if p_start_idx == 0: # Min prev_bal for prev_len > 0 must be 1
                    p_start_idx = 1
                
                p_end_idx = k_len - 1 # Max prev_bal for prev_len is k_len-1

                if p_start_idx > p_end_idx:
                    count_dp[k_len][bal] = 0
                else:
                    current_sum = row_prefix_sum_for_prev_len[p_end_idx]
                    if p_start_idx > 1: # check p_start_idx-1 for array access (sum up to p_start_idx-1)
                        current_sum = (current_sum - row_prefix_sum_for_prev_len[p_start_idx - 1] + MOD) % MOD
                    count_dp[k_len][bal] = current_sum
        
        # Update prefix sums for the row we just computed (count_dp[k_len])
        # This will be used for the next iteration (k_len+1)
        if k_len > 0 : # Balances are 1 to k_len for count_dp[k_len]
            row_prefix_sum_for_prev_len[0] = 0 # count_dp[k_len][0] is 0
            current_sum_val = 0
            for p_idx in range(1, k_len + 1):
                current_sum_val = (current_sum_val + count_dp[k_len][p_idx]) % MOD
                row_prefix_sum_for_prev_len[p_idx] = current_sum_val
            # Clear out sums for indices > k_len, if any were set by longer previous rows
            for p_idx in range(k_len + 1, N + 1):
                 row_prefix_sum_for_prev_len[p_idx] = current_sum_val


    ans = 0
    current_bal = 1 

    # Precompute prefix sums for all rows of count_dp for faster digit DP part
    # count_dp_prefix_sums[k_row][p_idx] = sum_{j=0..p_idx} count_dp[k_row][j]
    count_dp_prefix_sums = [[0] * (N + 1) for _ in range(N + 1)]
    for k_row in range(N + 1):
        if k_row == 0:
            count_dp_prefix_sums[0][0] = count_dp[0][0]
            # For j > 0, count_dp[0][j]=0, so sum is same.
            for j in range(1, N + 1):
                 count_dp_prefix_sums[0][j] = count_dp_prefix_sums[0][0]
        else:
            # count_dp[k_row][0] is 0 for k_row > 0
            count_dp_prefix_sums[k_row][0] = 0 
            current_sum_val = 0
            for p_idx in range(1, k_row + 1): # Iterate up to max bal for this row: k_row
                current_sum_val = (current_sum_val + count_dp[k_row][p_idx]) % MOD
                count_dp_prefix_sums[k_row][p_idx] = current_sum_val
            # For indices > k_row, sum is same as sum up to k_row
            for p_idx in range(k_row + 1, N + 1):
                 count_dp_prefix_sums[k_row][p_idx] = current_sum_val


    for i in range(N): 
        limit_val_exclusive = A[i] 
        
        # Try X_i = val, where val < limit_val_exclusive
        # Sum over val from 0 to limit_val_exclusive - 1
        if i == N - 1: # Last element
            # next_bal must be 0. So, current_bal + val - 1 == 0  => val = 1 - current_bal
            val_to_make_zero = 1 - current_bal
            if 0 <= val_to_make_zero < limit_val_exclusive:
                ans = (ans + 1) % MOD
        else: # Not the last element
            k_rem = N - (i + 1) # Length of suffix
            
            # target_bal = current_bal + val - 1
            # Sum count_dp[k_rem][target_bal] for val in [0, limit_val_exclusive-1]
            # Range of target_bal:
            # min_target_bal_for_sum = current_bal + 0 - 1 = current_bal - 1
            # max_target_bal_for_sum = current_bal + (limit_val_exclusive - 1) - 1 = current_bal + limit_val_exclusive - 2
            
            # Valid balances for count_dp[k_rem][target_bal] are [1, k_rem]
            # So effective range for target_bal is:
            # p_start = max(1, min_target_bal_for_sum)
            # p_end = min(k_rem, max_target_bal_for_sum)
            
            p_start = current_bal - 1
            if p_start < 1: p_start = 1 # Balances must be >0 for non-empty suffix

            p_end = current_bal + limit_val_exclusive - 2
            
            eff_p_start = max(1, p_start)
            eff_p_end = min(k_rem, p_end) # Max balance for length k_rem is k_rem

            if eff_p_start <= eff_p_end:
                # Sum count_dp[k_rem][p] for p from eff_p_start to eff_p_end
                terms_sum = count_dp_prefix_sums[k_rem][eff_p_end]
                if eff_p_start > 1: 
                    terms_sum = (terms_sum - count_dp_prefix_sums[k_rem][eff_p_start - 1] + MOD) % MOD
                ans = (ans + terms_sum) % MOD
        
        # Update current_bal for X_i = A[i] (path for sequence A itself)
        current_bal = current_bal + A[i] - 1

        if current_bal < 0: 
            break 
        if i < N - 1:
            if current_bal == 0: 
                break
            if current_bal > N - (i + 1): # Balance too high for remaining elements
                break
            
        if i == N - 1: 
            if current_bal == 0: 
                ans = (ans + 1) % MOD
    
    print(ans)

solve()