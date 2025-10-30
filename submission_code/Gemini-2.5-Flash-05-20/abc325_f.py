import math

def solve():
    N = int(input())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())

    # Re-index sensors so that K_A is the smaller K value.
    # This limits the DP state space based on min(K1, K2).
    if K1 <= K2:
        L_A, C_A, K_A = L1, C1, K1
        L_B, C_B, K_B = L2, C2, K2
    else:
        L_A, C_A, K_A = L2, C2, K2
        L_B, C_B, K_B = L1, C1, K1

    # dp[k_A_total] stores the minimum total k_B_total needed
    # to cover all sections processed so far, using exactly k_A_total sensors of type A.
    # Initialize with infinity, meaning unreachable states.
    dp = [float('inf')] * (K_A + 1)
    dp[0] = 0 # Base case: 0 Type A sensors, 0 Type B sensors needed for 0 sections covered.

    for d_i in D:
        next_dp = [float('inf')] * (K_A + 1)
        
        # Iterate through all possible previous states (k_A_prev_total, k_B_prev_total)
        for k_A_prev_total in range(K_A + 1):
            if dp[k_A_prev_total] == float('inf'):
                continue # This previous state is unreachable

            k_B_prev_total = dp[k_A_prev_total]

            # Calculate the remaining budget for Type B sensors
            remaining_K_B = K_B - k_B_prev_total
            
            # Determine the minimum k_A_for_d_i needed for the current section d_i
            # to ensure that the required k_B_for_d_i does not exceed `remaining_K_B`.
            # This comes from the inequality: d_i - k_A_for_d_i * L_A <= remaining_K_B * L_B
            # Rearranging: k_A_for_d_i * L_A >= d_i - remaining_K_B * L_B
            
            min_k_A_for_d_i_needed_for_k_B_limit = 0
            required_len_from_A_if_B_budget_maxed = d_i - remaining_K_B * L_B
            if required_len_from_A_if_B_budget_maxed > 0:
                min_k_A_for_d_i_needed_for_k_B_limit = math.ceil(required_len_from_A_if_B_budget_maxed / L_A)
            
            # Iterate through possible number of Type A sensors to use for the current section d_i.
            # The range is from `min_k_A_for_d_i_needed_for_k_B_limit`
            # up to the maximum allowed by the K_A budget (`K_A - k_A_prev_total`).
            start_k_A_for_d_i = min_k_A_for_d_i_needed_for_k_B_limit
            end_k_A_for_d_i = K_A - k_A_prev_total

            for k_A_for_d_i in range(start_k_A_for_d_i, end_k_A_for_d_i + 1):
                current_len_by_A = k_A_for_d_i * L_A
                rem_len = d_i - current_len_by_A
                
                k_B_for_d_i = 0
                if rem_len > 0:
                    k_B_for_d_i = math.ceil(rem_len / L_B)
                
                # The `min_k_A_for_d_i_needed_for_k_B_limit` calculation ensures
                # that `k_B_prev_total + k_B_for_d_i` will not exceed `K_B`.
                # So, no explicit check is needed here for `k_B_new_total > K_B`.

                k_A_new_total = k_A_prev_total + k_A_for_d_i
                k_B_new_total = k_B_prev_total + k_B_for_d_i

                # Update next_dp with the minimum k_B_total for this k_A_total
                next_dp[k_A_new_total] = min(next_dp[k_A_new_total], k_B_new_total)
        
        dp = next_dp # Update the DP table for the next section

    # After processing all sections, find the minimum total cost
    min_total_cost = float('inf')
    for k_A_total in range(K_A + 1):
        if dp[k_A_total] != float('inf') and dp[k_A_total] <= K_B:
            # If a valid combination of k_A_total and k_B_total is found within limits
            current_cost = k_A_total * C_A + dp[k_A_total] * C_B
            min_total_cost = min(min_total_cost, current_cost)

    if min_total_cost == float('inf'):
        print(-1) # Impossible to monitor all sections
    else:
        print(min_total_cost)

solve()