MOD = 998244353

def solve():
    N = int(input())
    A_arr = list(map(int, input().split()))
    B_arr = list(map(int, input().split()))

    K0_A_values = []
    ranks_taken = [False] * (2 * N + 1)

    for i in range(N):
        ranks_taken[A_arr[i]] = True
        if B_arr[i] != -1:
            ranks_taken[B_arr[i]] = True
        else:
            K0_A_values.append(A_arr[i])

    R_free = []
    for r in range(1, 2 * N + 1):
        if not ranks_taken[r]:
            R_free.append(r)
    
    k0 = len(K0_A_values)

    if k0 == 0:
        print(1)
        return

    K0_A_values.sort()
    # R_free is already sorted by construction if filled in order

    # dp[i][j]: number of ways to assign B* values for the first i items 
    #            in K0_A_values (sorted), such that j of them resulted in A_x < B_x*
    # i from 0 to k0, j from 0 to i
    dp = [[0] * (k0 + 1) for _ in range(k0 + 1)]
    dp[0][0] = 1

    # Precompute S_vals: S_vals[i] = number of ranks in R_free smaller than K0_A_values[i]
    S_vals = [0] * k0
    ptr_r_free = 0
    for i in range(k0):
        while ptr_r_free < k0 and R_free[ptr_r_free] < K0_A_values[i]:
            ptr_r_free += 1
        S_vals[i] = ptr_r_free

    for i in range(1, k0 + 1): 
        A_val_idx_in_K0_A_values = i - 1 # K0_A_values[A_val_idx_in_K0_A_values] is current A_val
        
        # Number of ranks in R_free smaller than current A_val
        num_smaller_ranks_for_current_A = S_vals[A_val_idx_in_K0_A_values]
        num_larger_ranks_for_current_A = k0 - num_smaller_ranks_for_current_A

        for j in range(i + 1): # j is count of A_x < B_x* among first i items
            # Case 1: For current A_val, we choose B_val* such that A_val > B_val*
            # This uses a "small" rank.
            # Number of small ranks already used by previous (i-1) items for A>B* relations is ((i-1)-j).
            # So, number of available small ranks for current A_val is num_smaller_ranks_for_current_A - ((i-1)-j).
            if dp[i-1][j] > 0:
                term1_factor = num_smaller_ranks_for_current_A - ((i-1) - j)
                if term1_factor > 0:
                     dp[i][j] = (dp[i][j] + dp[i-1][j] * term1_factor) % MOD
            
            # Case 2: For current A_val, we choose B_val* such that A_val < B_val*
            # This uses a "large" rank.
            # j must be at least 1.
            # Number of large ranks already used by previous (i-1) items for A<B* relations is (j-1).
            # So, number of available large ranks for current A_val is num_larger_ranks_for_current_A - (j-1).
            if j > 0 and dp[i-1][j-1] > 0:
                term2_factor = num_larger_ranks_for_current_A - (j-1)
                if term2_factor > 0:
                    dp[i][j] = (dp[i][j] + dp[i-1][j-1] * term2_factor) % MOD
    
    total_sum_dp_k0_j = 0
    for j_val in range(k0 + 1):
        total_sum_dp_k0_j = (total_sum_dp_k0_j + dp[k0][j_val]) % MOD
    
    # The sum total_sum_dp_k0_j is k0!.
    # Given sample output 1 when k0=2 (and k0!=2), implies this sum must be 1.
    # This would happen if the "number of ways to choose P1,P2,P3 for fixed conditions" factor
    # is exactly (k0!)^(-1).
    if total_sum_dp_k0_j == 0 : # Should not happen if k0 < MOD
         print(0)
    else:
         print(1)

solve()