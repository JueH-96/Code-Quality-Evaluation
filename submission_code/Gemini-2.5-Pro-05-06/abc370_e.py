import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353
    
    map_val_to_sum_dp = {0: 1} 
    total_sum_of_dp_vals_so_far = 1
    current_pref_sum = 0
    
    # ans_for_dp_N will store dp[i] for the prefix ending at A[i-1].
    # Since N >= 1, loop will run at least once.
    # It will be correctly updated to dp[N] by the end of the loop.
    # Initializing to dp[0]=1 handles N=0 case, but N >= 1 by constraints.
    ans_for_dp_N = 1 

    for idx_in_A in range(N):
        current_pref_sum += A[idx_in_A] 
        
        val_to_subtract = map_val_to_sum_dp.get(current_pref_sum - K, 0)
        
        current_dp_val = (total_sum_of_dp_vals_so_far - val_to_subtract + MOD) % MOD
        
        map_val_to_sum_dp[current_pref_sum] = \
            (map_val_to_sum_dp.get(current_pref_sum, 0) + current_dp_val) % MOD
            
        total_sum_of_dp_vals_so_far = (total_sum_of_dp_vals_so_far + current_dp_val) % MOD
        
        ans_for_dp_N = current_dp_val

    sys.stdout.write(str(ans_for_dp_N) + "
")

solve()