import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Max possible initial stones A_max_val = 5*10^5. Max N = 5*10^5.
    # Max B_intermediate[j] approx A_max_val + N approx 10^6.
    # Max dp_offset approx N approx 5*10^5.
    # Max nominal value stored in dp array = Max B_intermediate[j] + Max dp_offset 
    # approx 10^6 + 5*10^5 = 1.5 * 10^6.
    # Add a small buffer for safety.
    dp_array_size = 1500000 + 5 
    dp = [0] * dp_array_size

    dp_offset = 0
    
    B_intermediate = list(A) 
    nominal_values_at_addition = [0] * N

    num_adults_so_far = 0
    num_adults_with_zero_or_less_stones = 0 # Count of adults with Nominal_k <= dp_offset

    for i in range(N):
        num_givers = num_adults_so_far - num_adults_with_zero_or_less_stones
        
        B_intermediate[i] += num_givers
        
        current_nominal_value_for_i = B_intermediate[i] + dp_offset
        nominal_values_at_addition[i] = current_nominal_value_for_i
        
        # Ensure index is within bounds; practically, it should be due to dp_array_size calculation.
        # However, if B_intermediate[i] is negative (e.g. A[i] very small, num_givers 0),
        # current_nominal_value_for_i could be small. It won't be negative if A[i] >= 0.
        dp[current_nominal_value_for_i] += 1
        
        if current_nominal_value_for_i <= dp_offset: # True if B_intermediate[i] <= 0
            num_adults_with_zero_or_less_stones += 1
            
        num_adults_so_far += 1
        
        # Update count of non-givers due to dp_offset increment.
        # Adults whose nominal value was (dp_offset + 1) had 1 actual stone.
        # After dp_offset increments, they will have 0 actual stones.
        # Their count is dp[dp_offset + 1].
        # This must be done *before* dp_offset is actually changed.
        if dp_offset + 1 < dp_array_size: # Check bounds for dp[dp_offset + 1]
             num_adults_with_zero_or_less_stones += dp[dp_offset + 1]
        
        dp_offset += 1

    ans = [0] * N
    final_dp_offset = N # dp_offset increments N times from 0, so it becomes N.
    for i in range(N):
        ans[i] = max(0, nominal_values_at_addition[i] - final_dp_offset)
        
    sys.stdout.write(" ".join(map(str, ans)) + "
")

solve()