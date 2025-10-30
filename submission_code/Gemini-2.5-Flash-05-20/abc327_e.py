import math

def solve():
    N = int(input())
    P = list(map(int, input().split()))

    # dp[i][k] stores the maximum weighted sum of performances (W_Q)
    # for a sequence of 'k' contests where P[i] is the k-th (last) chosen contest.
    # P uses 0-based indexing, so P[i] is the (i+1)-th original contest.
    # k ranges from 1 to N.
    # dp table dimensions: N rows (for i from 0 to N-1), N+1 columns (for k from 1 to N).
    dp = [[-float('inf')] * (N + 1) for _ in range(N)]

    # Initialize overall_max_rating to a very small number, as ratings can be negative.
    overall_max_rating = -float('inf')

    # Precompute powers of 0.9 and square roots for efficiency.
    # powers_of_0_9[k_val] stores (0.9)^k_val
    powers_of_0_9 = [1.0] * (N + 1) 
    for k_val in range(1, N + 1):
        powers_of_0_9[k_val] = powers_of_0_9[k_val - 1] * 0.9

    # sqrt_k_values[k_val] stores sqrt(k_val)
    sqrt_k_values = [0.0] * (N + 1) 
    for k_val in range(1, N + 1):
        sqrt_k_values[k_val] = math.sqrt(float(k_val)) # Use float for math.sqrt input

    # Base case: k = 1 (choosing a single contest P[i])
    for i in range(N):
        dp[i][1] = float(P[i]) # W_Q for k=1 is just P[i]
        
        # Calculate rating for k=1: R = P[i] / (10*(1-0.9^1)) - 1200 / sqrt(1)
        # R = P[i] / 1.0 - 1200.0 / 1.0 = P[i] - 1200.0
        current_rating = dp[i][1] - 1200.0
        if current_rating > overall_max_rating:
            overall_max_rating = current_rating

    # Fill DP table for k from 2 to N
    for k in range(2, N + 1):
        # max_prev_dp_val stores the maximum dp[prev_idx][k-1] encountered so far for prev_idx < i.
        # This is needed to calculate dp[i][k].
        max_prev_dp_val = -float('inf') 
        
        for i in range(N):
            # A sequence of 'k' contests ending at P[i] implies that P[i] is the k-th contest.
            # This means we must have selected 'k-1' contests from P[0]...P[i-1].
            # This is only possible if there are at least 'k-1' contests before P[i],
            # i.e., i (0-based index) must be at least k-1.
            if i >= k - 1:
                # If there was at least one valid previous state for k-1 contests (max_prev_dp_val is not -inf)
                if max_prev_dp_val != -float('inf'):
                    # The recurrence: dp[i][k] = 0.9 * (max_dp_value_from_previous_k_minus_1_states_before_i) + P[i]
                    dp[i][k] = 0.9 * max_prev_dp_val + P[i]
            
            # Update max_prev_dp_val for the *next* iteration of 'i'.
            # It should include the current dp[i][k-1] for the calculation of dp[i+1][k].
            # We only consider valid dp[i][k-1] values for this max.
            if dp[i][k-1] != -float('inf'):
                max_prev_dp_val = max(max_prev_dp_val, dp[i][k-1])

            # If dp[i][k] is a valid value (i.e., it was computed and not -inf)
            if dp[i][k] != -float('inf'):
                current_W_Q = dp[i][k]
                # W_C = sum_{j=0}^{k-1} (0.9)^j = (1 - (0.9)^k) / (1 - 0.9) = 10 * (1 - (0.9)^k)
                current_W_C = 10.0 * (1.0 - powers_of_0_9[k])
                
                # Calculate rating: R = W_Q / W_C - 1200 / sqrt(k)
                current_rating = current_W_Q / current_W_C - 1200.0 / sqrt_k_values[k]
                
                if current_rating > overall_max_rating:
                    overall_max_rating = current_rating

    # Print the maximum possible rating formatted to 15 decimal places.
    print(f"{overall_max_rating:.15f}")

# Call the solve function to run the program
solve()